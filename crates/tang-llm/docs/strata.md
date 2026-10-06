# How Strata runs Qwen3.8-Flash-Next fast

Notes from reading [Strata](https://github.com/Niko1221/Strata) (MIT, C++/CUDA, read at `6f32ec0`,
2026-10-04): its docs, its paper (`docs/paper/Strata-Paper.pdf`) and its source. This is the
reference for [hybrid-moe.md](hybrid-moe.md). Paths are into the Strata tree; `gen` is
`src/program/generate.cpp`. Numbers are Strata's own, mostly on an RTX 5070 12 GB + Ryzen 5 7600
(AVX-512) + 64 GB DDR5-5200, unless marked as measured on mew.

## The model, as the engine sees it

GGUF architecture `qwen4exp`. 48 layers; 2560 wide; residual is 4 streams (hyper-connections).

- **Mixers.** Layers `l % 4 == 3` (12 of them) are QSA sparse attention; the other 36 are Gated
  DeltaNet.
- **FFN.** Every layer is MoE: 512 experts, top-10 by softmax over all 512, renormalised, plus a
  shared expert with a sigmoid scalar gate. Expert FFN is 640 wide.
- **N-gram embedding (PLE).** One extra block before layer 1 (0-indexed; HF's `ple_layer_ids [2]`
  appears to be 1-indexed). It reads 16 rows of a 320M-row table hashed from the last 3 tokens.
- **MTP.** One full-attention layer that drafts the next tokens.

### Block math

From Strata's kernel headers and parity tests (`include/strata/kernels/*.hpp`, `src/core/layer.cpp`).
Strata's `ref/*.py` references are not in the repo; ground truth is llama.cpp `3cf03257`.

```
R[c] = embed(tok) for c in 0..3                       # 4 streams × 2560
for l in 0..47:
  if l == 1: R = ple_block(R)
  x, inj = hc_read(R, hc_attn_*);   R = hc_write(R, (l%4==3 ? qsa : gdn)(x), inj)
  x, inj = hc_read(R, hc_ffn_*);    R = hc_write(R, moe(x), inj)
logits = output @ hc_read(R, output_hc_*, no inject)  # no separate final norm

hc_read(R):  xn[c] = rmsnorm(R[c]) * w_norm[c]        # per stream; w_norm stored as 1+w
             gate  = sigmoid(up[10240×320] @ silu((down[320×10240] @ xn) / 4))
             x     = mean_c(xn[c] * gate[c]);  inj = inject[4×10240] @ xn
hc_write:    R[c] += y * 2*sigmoid(inj[c] / 4)

gdn(x):  qkv = W_qkv @ x                  # [q 2048 | k 2048 | v 6144]
         h = silu(causal_conv4(qkv))      # all 10240 channels, before the split
         q,k (16 heads), v (48 heads);  q,k /= sqrt(Σx² + eps)   # eps on the squared norm
         beta = sigmoid(W_b @ x);  g = exp(softplus(W_a @ x + dt_bias) * -exp(A_log))
         per v head h, s = h % 16:  S = g·S;  S += k[s] ⊗ beta·(v[h] − Sᵀk[s]);  o = Sᵀq[s] / √128
         y = rmsnorm_head(o) * ssm_norm * sigmoid(W_z @ x);  out = W_o @ y

qsa(x):  [q | gate] = W_q @ x per head (halves, 256 each)
         q = rope64(rmsnorm(q)·qn);  k = rope64(rmsnorm(W_k @ x)·kn);  v = W_v @ x   # neox, θ 1e7
         indexer: raw = W_ik @ x (128); every 4th cell pooled[b] = rope(rmsnorm(mean raw[4b..4b+3])·n, 4b)
                  qi = rope(rmsnorm(W_iq @ x)·n) (4 heads);  score[b] = Σ_h relu(qi[h]·pooled[b])
                  tail block +1e9;  pick top 2051 cells (blocks weighted by cell count), ascending
         o = softmax(q·k/16)·v over picked cells;  out = W_o @ (o * sigmoid(gate))

moe(x):  p = softmax(W_r @ x) over 512;  top-10, w = p / max(Σp, 2^-14)
         y = Σ w_i · down_i(silu(gate_i x) * up_i x)  +  sigmoid(w_sg·x) · shared(x)

ple:     16 rows = hash(tok, prev1, prev2) → 2560;  key = gnorm(W_k @ e), q = gnorm(R)
         gate[c] = sigmoid(sign(s)·√|s|), s = <key[c], q[c]>/√2560;  gated = (W_v @ e)·gate
         R += gated + silu(depthwise conv k=4, dilation 3 over 9 rows of gnorm(gated) history)
```

The traps worth repeating:

- **GDN head pairing is modulo** in GGUF order (`v head h` uses `k head h % 16`); HF order
  probably needs `h / 3`.
- **GDN L2 eps** floors the *squared* norm.
- **GDN state** decays before the delta update, and the readout uses the updated state. State and
  conv history are fp32.
- **QSA indexer:** pool the 4 raw keys, *then* norm, then rope at the block's first cell. Score
  = Σ_heads relu(q·k), no softmax, no scale. Always select the incomplete tail block. Width
  2051 cells, not 2048. KV head = q_head / 12.
- **Hyper-connection read:** per-stream RMS, `silu(down·x / hc)`, `sigmoid(up·lo)`, *mean* over
  streams. **Write:** `R[c] += y · 2σ(inject[c] / hc)`. The `hc_*_norm` weights are stored as
  `1 + w` already.
- **Router input precision matters.** An 8e-3 error in the logits flips top-10 membership.

## Where the bytes live

| Part | Size (Q2_0 file) | Read per token | Lives in |
|---|---|---|---|
| Routed experts, 48 × 512 | 34 GB | 0.66 GB (2%) | pinned RAM; the hottest ~50–90% of *traffic* cached in VRAM |
| Mixers, routers, shared experts | 1.8 GB | all | VRAM |
| Hyper-connection weights | 1.3 GB | all | VRAM (bf16) |
| Output head | 0.44 GB | all | VRAM |
| MTP layer (all 512 of its experts) | 0.8 GB | per draft | VRAM |
| N-gram table | 28.8 GB (IQ4_NL) | 16 rows, ≤64 KiB of 4K pages | SSD, `O_DIRECT`, never the page cache |
| GDN state | 118 MB fixed | all | VRAM |

The expert slot cache is sized last, from `cudaMemGetInfo` after everything else is allocated
(`gen:3198-3420`).

- **Slot granularity.** A slot is one (layer, expert) blob: 1,382,400 B for Q2_0, or the GGUF's
  per-layer size for i-quants (byte-sized slots gave +13%).
- **Seeding.** The cache starts from a shipped frequency profile of all 24,576 pairs
  (`data/expert-profile.bin`).
- **Adapting.** Every 4 rounds a thread scores `usage` counts:
  - it swaps up to 96 experts, where a candidate's count is ≥ 2 and beats its victim by 1.5;
  - it then decays all counts by 0.7.
- **Swap mechanics.** The victim goes non-resident immediately. The copy runs on a side stream,
  and the new expert is admitted only when the copy lands.
- **Hit rate.** Profile alone gives ~0.50 of routed experts from 4,500 slots. Adaptive swapping
  raises that to 0.72 (12 GB), ~0.93 (24 GB).

The host arena is anonymous hugepage memory, registered with
`cudaHostRegister(Portable | Mapped)` *before first touch*. If one range is refused it falls
back to per-layer slices, then to `mlock`.

- Experts served from a file mmap ran at 19 GB/s, against 42.8 GB/s from the arena. The CPU
  path never reads file pages.
- **Low-RAM mode** (`--resident-experts`) keeps in RAM only the complement of what VRAM holds.

## A decode step is a verify window

There is no one-token decode loop in production. Every step runs T = 1..8 tokens through all 48
layers: the last accepted token plus up to 3 MTP drafts, or 5 from prompt lookup. This is one
pre-captured CUDA graph per T (`include/strata/core/verify.hpp:300`).

- **Dense weights** are read once for all T (multi-column MMVQ, bit-identical to T = 1).
- **Experts.** Each distinct missed expert is read once for every token routed to it. A
  window's misses are 1.75× / 2.4× / 3.05× one token's at T = 2 / 3 / 4.

Per layer, inside the graph:

```
GPU: hc read → mixer → hc read → router top-10 for T tokens
     → doorbell: ids (+ activations if anything misses) into mapped host memory, fence, seq++
     ┊ fork: shared expert on a side branch
host (spinning on seq): count usage, dedupe, classify each expert:
     VRAM slot | PCIe share | CPU
     → publish the GPU's plan (flag A) FIRST, then start CPU work
GPU: wait flag A → grouped expert kernel over VRAM slots
     → copy kernel pulls the PCIe share (last 20–55% of misses) from the mapped arena
     → grouped kernel on those
CPU: all physical cores, row-split: gate/up rows → quantize → down rows
     → fp32 rows into mapped memory, flag B
GPU: wait flag B → copy only CPU rows → add → combine Σ wᵢ·yᵢ + shared → next layer
```

How the window is wired:
- **No host callbacks, no per-layer events.** The waits are single-thread spin kernels on
  mapped flags. The host polls `cudaStreamQuery` every 2 ms only to flush and to notice a dead
  graph.
- **One `cudaStreamSynchronize` per window.**
- **Router weights are applied only on the GPU.**
- **The commit graph (below) runs async.** The MTP draft and the cache-adapt thread overlap it.

Window time for Q2_0 at 4K (paper Table 5): 34.3 ms = 14.6 GPU + 13.4 CPU experts + 2.0 draft +
4.3 other, for 3.23 tokens. The GPU half is mostly reading 3.5 GB of dense weights.

## CPU experts

The pool and its kernels:
- **Workers.** One pinned worker per physical core except the host's. The host thread also
  works.
- **Claiming and parking.** Workers claim jobs with a CAS on one `epoch|njobs|idx` word. They
  spin for 20 ms, then sleep.
- **Row split.** Each phase (gate/up, then down) is split into `3 × threads` row ranges *across
  all experts*, so every core streams DRAM even when only 2–3 experts miss.
- **Q2_0 kernel.** AVX-512 VNNI + VBMI. One unpack and one `vpdpbusd` per 64 weights; the
  `Σ(c−1)·x = Σc·x − Σx` correction is applied once per chunk. It runs at ~42 GB/s on 6 cores,
  i.e. RAM-bound.
- **I-quants** (IQ2_XS, IQ3_*) are compute-bound on codebook decode, ~5 GB/s per core. Their
  kernels decode a 32/64-value chunk once and apply it to every token in the window.
- **Activation quantisation.** 32-value chunks, `amax / 127`, round half away from zero. The GPU
  uses the same fp32 scales so hits and misses agree.

## Speculation and the recurrent state

- **MTP drafting.** MTP drafts a chain (no trees), conditioned on the main model's final
  4-stream residual. It attends densely over a 32K ring and touches no indexer state. A draft is
  added only while its probability is ≥ 0.5.
- **Prompt-lookup drafting.** A 3-gram hash over prompt and output, min match 3, up to 5 drafts.
  It is used only when its first token agrees with the MTP's first draft *and* an online cost
  model (EMA of measured ms per T, decayed acceptance per match length) says it wins
  (`src/spec/draft_policy.cpp`).
- **Acceptance is exact match.** Greedy, or sampled with `Philox(seed, position)`, so output is a
  function of the seed regardless of drafts. There is no p/q rejection sampling.

GDN rollback is the crux of the whole design. **The verify window never writes the recurrent
state.**
- **During the window**, each GDN layer stores its per-token inputs (conv input, post-conv
  q|k|v, gate, beta) in scratch. The step kernel walks the T tokens in registers from the old
  state and only emits outputs.
- **A separate commit graph** replays the recurrence for the `n_keep` accepted tokens and
  writes the state. It also sets the conv history to the last 3 of `[history | accepted
  inputs]`. `n_keep` lives in device memory, so one graph serves every accept count.
- **What it costs:** no state copies, and O(n_keep) recurrence steps.
- **Positional state is simply overwritten.** QSA K/V and MTP K/V cells get rewritten before
  anything reads them.
- **Indexer tail** (raw keys of the unfinished block): snapshot per window, restore, re-append
  the accepted keys.
- **PLE history:** a snapshot per row.

Measured gain from speculation: 1.6–1.8×, from 47–57 to 82–92 tok/s at 4K. That is less than the
2.5–3× of all-VRAM engines, because each extra token routes to its own CPU experts.

## Prefill

- **Streaming.** From 1,024 tokens nearly every expert is routed, so the engine streams *every*
  non-resident expert in fixed (layer, id) order through a ring. Copies run ahead across layers,
  so the next layer's experts arrive during its attention.
- **Buffers.** Chunk buffers borrow the top expert-cache slots and refill them afterwards. Chunk
  size (≤ 8192) and ring share one byte budget.
- **GDN** walks the chunk sequentially in one kernel. It is not chunked WY.
- **GEMMs** use int8 tensor cores (llama.cpp MMQ). QSA prompt attention uses `mma` f16.
- **Result:** Q2_0 at 32K: 572 → 2,653 tok/s over 0.1.13–0.1.36.

## State, KV, conversations

- **Two kinds of state.**
  - *Positional:* QSA K/V, pooled indexer keys, MTP K/V. Rewound by position, never copied.
  - *Running:* GDN state + conv, indexer tails, PLE history. About 118 MB, and it is the whole
    of a checkpoint.
- **Checkpoints** are taken at the last turn boundary, at a ≥2K system prompt (pinned), and every
  16K tokens. Up to 6 are kept, in RAM only. Follow-ups start in about 0.5 s.
- **KV format.** int8 above 8K. Q4 with Hadamard rotation is opt-in.
- **KV streaming above 64K.** Authoritative K/V lives in mapped host RAM. A capturable kernel
  makes the selected blocks resident (CLOCK) after QSA selection, before attention.

## What didn't work for them

- Splitting a window into two token groups to overlap CPU and GPU: −7% (dense weights get read
  twice).
- Copying more than ~20% of misses on the critical path.
- Blocking cache refills.
- cuBLAS grouped GEMM for prefill experts.
- Helper GPUs.
- Host-callback DMA for the PCIe share: it deadlocked on a driver lock (issue #31), and was
  replaced by an in-graph copy kernel.

## Baseline on mew (measured 2026-10-04)

- **llama.cpp** (`server-cuda` b11382), unsloth UD-Q2_K_XL, `-ngl 99 --n-cpu-moe 35 -fa on -c
  32768`:
  - 19.7 GB VRAM;
  - **20–22 tok/s** warm decode;
  - `--n-cpu-moe 30` OOMs.
- **Strata on mew:** not measured yet. Its README *estimates* 100–140 tok/s on a 3090.
