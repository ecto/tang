# Tiled prefill attention: next steps

The current `cx_attn_tile` path stages K/V in shared memory and reuses each tile across
eight query rows. The measured baseline for wide prefill without tiling is 4.7 s at 8k,
28.5 s at 30k, and 72 s at 54k. The 30k target is about 10 s. First run
`scripts/bench_prefill_tile.sh` on an H200 with the real model; require greedy identity for
64 steps and teacher-forced max KL no greater than `1e-3` against narrow attention.

## 1. FA2-style tensor-core attention

Replace the scalar score/value work in the tiled kernel with an `mma.sync` design using
Br64/Bc32. Swizzle shared-memory K/V layouts and use `cp.async` to stage the next K/V tile
while the current tile is consumed. Preserve the causal mask, online softmax accumulation,
GQA mapping, and output gate. Benchmark register use, occupancy, achieved bandwidth, and
the correctness gate above before changing the default. FlashAttention (FA) is BSD-3-Clause;
review the exact source and preserve required notices if adapting implementation code.

## 2. Gated DeltaNet chunked prefill

After attention is measured and stable, implement chunked prefill for Gated DeltaNet using
the `fla` approach (MIT). Profile its convolution and recurrent state update separately,
then evaluate chunk sizes and state handoff. Require the existing narrow-path logits and
greedy checks plus teacher-forced KL no greater than `1e-3` before enabling it by default.
