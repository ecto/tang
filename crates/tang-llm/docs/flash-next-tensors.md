# Qwen3.8-Flash-Next: what the GGUF holds

Measured 2026-10-04 on mew with `tang-llm gguf-info` (the reader in `src/gguf.rs`), for the two
files we have:

- **ISTA** — `ISTA-DASLab` GSQ-RCO `Q2_0`, 2 shards, 66.41 GB of tensor data
  (`~/models/qwen3.8-flash-next-ista/Q2_0/Qwen3.8-Flash-Next-GSQ-RCO-Q2_0-0000{1,2}-of-00002.gguf`).
- **unsloth** — `UD-Q2_K_XL`, 3 shards, 78.86 GB
  (`~/models/qwen3.8-flash-next/UD-Q2_K_XL/…-0000{1,2,3}-of-00003.gguf`).

The math these tensors feed is in [strata.md](strata.md), as corrected at the bottom of this file.
`src/flash/reference.rs` is the executable version.

## Conventions

- **Dims are ggml `ne`, fastest first.** A projection `y = W x` with `x` of width `in` is stored
  `[in, out]`: row `o` is `in` contiguous elements. A fused expert tensor is `[in, out, 512]` and
  expert `e` is one contiguous run (`gguf::Gguf::expert`).
- Every quantised type packs whole blocks along `ne[0]`, so a row, and an expert, is a byte range.
- `N` is the layer, 0..47. Layer `l` is QSA when `l % 4 == 3` (12 layers:
  3, 7, …, 47; `attention.compress_ratios` is 4 there and 0 elsewhere) and GDN otherwise
  (36 layers). There is no `attention.recurrent_layers` key; llama.cpp derives it from
  `full_attention_interval = 4`.
- **No MTP block** in either file (`nextn` tensors absent, no `nextn_predict_layers` key). MTP comes
  from unsloth's separate `MTP/mtp-Qwen3.8-Flash-Next-Q8_0.gguf` (not on mew yet) or the HF
  checkpoint.

## Shards

| File | Tensors | Tensor bytes | Data starts at |
|---|---|---|---|
| ISTA `-00001-of-00002` | 1223: everything except the n-gram table | 37,612,715,520 | 11,024,672 |
| ISTA `-00002-of-00002` | 1: `per_layer_token_embd.weight` | 28,800,138,240 | 192 |
| unsloth `-00001-of-00003` | 0: metadata and tokenizer only | 0 | 10,946,624 |
| unsloth `-00002-of-00003` | 503: `output*`, `per_layer_token_embd`, `token_embd`, blk 0..19 (partly) | 49,979,747,456 | 31,840 |
| unsloth `-00003-of-00003` | 721: rest of blk 19..47 | 28,878,356,864 | 46,080 |

A reader that opens only the first shard sees no table (ISTA) or no tensors at all (unsloth);
`gguf::discover_shards` opens the set from any one name.

## Tensors

`count` is how many layers carry the tensor. Types are listed with how many layers use each:
**ISTA quantises per layer, not per role**, so a kernel can't assume a type from a name (e.g.
`attn_qkv` is IQ4_XS in 13 layers, Q3_K in 18, Q4_K in 4, Q2_0 in 1).

| Pattern | count | layers | ne | ISTA types | ISTA bytes | unsloth types | unsloth bytes | Role |
|---|---|---|---|---|---|---|---|---|
| `token_embd.weight` | 1 | — | `[2560, 248320]` | Q3_K | 273,152,000 | Q5_K | 437,043,200 | token embedding (row = token) |
| `per_layer_token_embd.weight` | 1 | — | `[160, 320001536]` | IQ4_NL | 28,800,138,240 | IQ4_NL | 28,800,138,240 | PLE n-gram table, 90 B per 160-wide row |
| `blk.N.hc_attn_norm.weight` | 48 | all | `[10240]` | F32 | 1,966,080 | F32 | 1,966,080 | HC before the mixer: per-stream RMSNorm gamma, stored as 1+w, read as `[2560, 4]` |
| `blk.N.hc_attn_down.weight` | 48 | all | `[10240, 320]` | BF16 | 314,572,800 | Q8_0 | 167,116,800 | HC: bottleneck down (10240 → 320) |
| `blk.N.hc_attn_up.weight` | 48 | all | `[320, 10240]` | BF16 | 314,572,800 | Q8_0 | 167,116,800 | HC: bottleneck up (320 → 10240) |
| `blk.N.hc_attn_inject.weight` | 48 | all | `[10240, 4]` | BF16 | 3,932,160 | F32 | 7,864,320 | HC: write-back logits (10240 → 4) |
| `blk.N.hc_ffn_{norm,down,up,inject}.weight` | 48 | all | as above | as above | 634,443,840 | as above | 344,063,680 | HC before the MoE, same four tensors |
| `blk.N.attn_qkv.weight` | 36 | GDN | `[2560, 10240]` | IQ4_XS×13 Q2_0×1 Q3_K×18 Q4_K×4 | 450,150,400 | Q5_K×35 Q6_K×1 | 652,288,000 | GDN q\|k\|v (2048\|2048\|6144) |
| `blk.N.attn_gate.weight` | 36 | GDN | `[2560, 6144]` | IQ4_XS×4 Q2_0×4 Q3_K×25 Q4_K×3 | 246,620,160 | Q5_K×35 Q6_K×1 | 391,372,800 | GDN output gate z |
| `blk.N.ssm_alpha.weight` | 36 | GDN | `[2560, 48]` | BF16 | 8,847,360 | F32 | 17,694,720 | GDN decay-gate projection (one per v head) |
| `blk.N.ssm_beta.weight` | 36 | GDN | `[2560, 48]` | BF16 | 8,847,360 | F32 | 17,694,720 | GDN beta projection |
| `blk.N.ssm_conv1d.weight` | 36 | GDN | `[4, 10240]` | F32 | 5,898,240 | F32 | 5,898,240 | GDN depthwise conv; tap `i` of channel `c` at `c*4 + i`, tap 0 = oldest |
| `blk.N.ssm_dt.bias` | 36 | GDN | `[48]` | F32 | 6,912 | F32 | 6,912 | GDN dt bias |
| `blk.N.ssm_a` | 36 | GDN | `[48]` | F32 | 6,912 | F32 | 6,912 | GDN `-exp(A_log)`, already negated and exponentiated |
| `blk.N.ssm_norm.weight` | 36 | GDN | `[128]` | F32 | 18,432 | F32 | 18,432 | GDN per-head output RMSNorm gamma (not 1+w) |
| `blk.N.ssm_out.weight` | 36 | GDN | `[6144, 2560]` | IQ4_XS×17 Q3_K×3 Q4_K×11 Q5_K×3 Q6_K×2 | 317,890,560 | Q6_K | 464,486,400 | GDN output projection |
| `blk.N.attn_q.weight` | 12 | QSA | `[2560, 12288]` | IQ4_XS×4 Q2_0×2 Q3_K×6 | 165,642,240 | Q5_K | 259,522,560 | QSA: per head `[q 256 \| gate 256]`, ×24 |
| `blk.N.attn_k.weight` | 12 | QSA | `[2560, 512]` | IQ4_XS×4 Q3_K×3 Q4_K×2 Q5_K×2 Q6_K×1 | 8,826,880 | Q6_K | 12,902,400 | QSA k, 2 kv heads × 256 |
| `blk.N.attn_v.weight` | 12 | QSA | `[2560, 512]` | IQ4_XS×3 Q3_K×1 Q4_K×2 Q5_K×3 Q6_K×3 | 10,055,680 | Q6_K | 12,902,400 | QSA v, 2 kv heads × 256 |
| `blk.N.attn_q_norm.weight` | 12 | QSA | `[256]` | F32 | 12,288 | F32 | 12,288 | QSA q RMSNorm gamma |
| `blk.N.attn_k_norm.weight` | 12 | QSA | `[256]` | F32 | 12,288 | F32 | 12,288 | QSA k RMSNorm gamma |
| `blk.N.attn_output.weight` | 12 | QSA | `[6144, 2560]` | Q4_K×4 Q5_K×4 Q6_K×4 | 130,252,800 | Q5_K | 129,761,280 | QSA output projection |
| `blk.N.indexer.k_proj.weight` | 12 | QSA | `[2560, 128]` | BF16 | 7,864,320 | BF16 | 7,864,320 | indexer raw key (pooled over 4 cells, then normed) |
| `blk.N.indexer.q_proj.weight` | 12 | QSA | `[2560, 512]` | BF16 | 31,457,280 | BF16 | 31,457,280 | indexer query, 4 heads × 128 |
| `blk.N.indexer.k_norm.weight` | 12 | QSA | `[128]` | F32 | 6,144 | F32 | 6,144 | indexer pooled-key RMSNorm gamma |
| `blk.N.indexer.q_norm.weight` | 12 | QSA | `[128]` | F32 | 6,144 | F32 | 6,144 | indexer query RMSNorm gamma |
| `blk.N.ffn_gate_inp.weight` | 48 | all | `[2560, 512]` | BF16 | 125,829,120 | F32 | 251,658,240 | MoE router |
| `blk.N.ffn_gate_exps.weight` | 48 | all | `[2560, 640, 512]` | Q2_0 | 11,324,620,800 | IQ2_XS×47 IQ3_XXS×1 | 11,717,836,800 | routed experts, gate |
| `blk.N.ffn_up_exps.weight` | 48 | all | `[2560, 640, 512]` | Q2_0 | 11,324,620,800 | IQ2_XS×47 IQ3_XXS×1 | 11,717,836,800 | routed experts, up |
| `blk.N.ffn_down_exps.weight` | 48 | all | `[640, 2560, 512]` | Q2_0 | 11,324,620,800 | IQ4_NL | 22,649,241,600 | routed experts, down |
| `blk.N.ffn_gate_inp_shexp.weight` | 48 | all | `[2560]` | BF16 | 245,760 | F32 | 491,520 | shared-expert scalar gate (sigmoid) |
| `blk.N.ffn_gate_shexp.weight` | 48 | all | `[2560, 640]` | IQ4_XS×6 Q2_0×21 Q3_K×13 Q4_K×6 Q5_K×1 Q6_K×1 | 32,051,200 | Q5_K×47 Q6_K×1 | 54,284,800 | shared expert, gate |
| `blk.N.ffn_up_shexp.weight` | 48 | all | `[2560, 640]` | IQ4_XS×5 Q2_0×13 Q3_K×21 Q4_K×6 Q5_K×2 Q6_K×1 | 34,252,800 | Q5_K×47 Q6_K×1 | 54,284,800 | shared expert, up |
| `blk.N.ffn_down_shexp.weight` | 48 | all | `[640, 2560]` | IQ4_NL×7 Q2_0×16 Q4_0×16 Q5_0×7 Q8_0×2 | 39,936,000 | Q8_0 | 83,558,400 | shared expert, down |
| `blk.1.ple_key.weight` | 1 | 1 | `[2560, 10240]` | Q2_0 | 7,372,800 | Q8_0 | 27,852,800 | PLE key (2560 → 4 streams × 2560) |
| `blk.1.ple_value.weight` | 1 | 1 | `[2560, 2560]` | BF16 | 13,107,200 | Q8_0 | 6,963,200 | PLE value |
| `blk.1.ple_norm_{key,query,conv}.weight` | 1 each | 1 | `[10240]` | F32 | 3 × 40,960 | F32 | 3 × 40,960 | PLE grouped RMSNorm gammas, read as `[2560, 4]`, not 1+w |
| `blk.1.ple_conv1d.weight` | 1 | 1 | `[4, 10240]` | F16 | 81,920 | F32 | 163,840 | PLE depthwise conv; tap `k` of channel `c` at `k + 4*c` |
| `output_hc_norm.weight` | 1 | — | `[10240]` | F32 | 40,960 | F32 | 40,960 | final HC read (it is the output norm), 1+w |
| `output_hc_down.weight` | 1 | — | `[10240, 320]` | BF16 | 6,553,600 | Q8_0 | 3,481,600 | final HC bottleneck down |
| `output_hc_up.weight` | 1 | — | `[320, 10240]` | BF16 | 6,553,600 | Q8_0 | 3,481,600 | final HC bottleneck up |
| `output.weight` | 1 | — | `[2560, 248320]` | Q5_K | 437,043,200 | Q4_K | 357,580,800 | LM head (not tied) |

There's no `output_norm`, no `output_hc_inject`, and no per-layer `attn_norm`/`ffn_norm`: the
hyper-connection reads are the norms.

One routed expert, `[gate | up | down]`: **ISTA 1,382,400 B** (3 × 460,800 Q2_0);
**unsloth 1,868,800 B** (2 × 473,600 IQ2_XS + 921,600 IQ4_NL; layer 2's IQ3_XXS gate/up make it
2,176,000 B there).

## Bytes by category

`gguf-info`'s buckets: *dense* is the mixers, routers, shared experts and the PLE block;
*head* is `output.weight` plus `output_hc_*`.

| Category | ISTA Q2_0 | unsloth UD-Q2_K_XL |
|---|---|---|
| dense (mixers, routers, shared experts, PLE block) | 1,645,422,080 (1.532 GiB) | 2,483,294,720 (2.313 GiB) |
| hyper-connections (`blk.N.hc_*`) | 1,270,087,680 (1.183 GiB) | 688,128,000 (0.641 GiB) |
| token embedding | 273,152,000 (0.254 GiB) | 437,043,200 (0.407 GiB) |
| head (`output.weight` + `output_hc_*`) | 450,191,360 (0.419 GiB) | 364,584,960 (0.340 GiB) |
| **routed experts** | **33,973,862,400 (31.641 GiB)** | 46,084,915,200 (42.920 GiB) |
| **n-gram table** | **28,800,138,240 (26.822 GiB)** | 28,800,138,240 (26.822 GiB) |
| total | 66,412,853,760 (61.852 GiB) | 78,858,104,320 (73.442 GiB) |

Dense + HC + head, everything a decode step reads in full: ISTA 3.37 GB, unsloth 3.54 GB. Plus
one embedding row, 16 table rows (1,440 B) and 480 experts (ISTA: 663.6 MB).

By type, ISTA: Q2_0 202 tensors 34.05 GB (31.71 GiB); IQ4_NL 8, 28.81 GB (the table and 7
shared-expert downs); BF16 483, 1.475 GB; Q3_K 91, 772 MB; Q5_K 16, 521 MB; IQ4_XS 56, 438 MB;
Q4_K 38, 232 MB; Q6_K 12, 84 MB; Q4_0 16, 14.7 MB; F32 292, 10.1 MB; Q5_0 7, 7.9 MB; Q8_0 2,
3.5 MB; F16 1, 82 kB.

Every type in both files has a dequantizer in `src/gguf.rs`, bit-exact against gguf-py on the
first rows of one tensor per type (`scripts/flash_gguf_check.py`: BF16 F16 F32 IQ2_XS IQ3_XXS IQ4_NL
IQ4_XS Q3_K Q4_0 Q4_K Q5_0 Q5_K Q6_K Q8_0, max |diff| = 0). gguf-py has no type 42; Q2_0 is checked
by a unit test against ggml's `dequantize_row_q2_0` and end to end by parity below.

## Metadata (`qwen4exp.*`, identical in both files)

| Key | Value |
|---|---|
| `block_count` | 48 |
| `embedding_length` | 2560 |
| `context_length` | 262144 |
| `attention.head_count` / `head_count_kv` | 24 / 2 |
| `attention.key_length` / `value_length` | 256 / 256 |
| `attention.layer_norm_rms_epsilon` | 1e-6 (every RMSNorm, the GDN L2 norm and the HC norms) |
| `attention.compress_ratios` | 48 entries, 4 at `l % 4 == 3`, else 0 |
| `attention.indexer.head_count` / `key_length` / `top_k` | 4 / 128 / 2048 |
| `full_attention_interval` | 4 |
| `rope.dimension_count` | 64 (of 256; indexer: 64 of 128) |
| `rope.dimension_sections` | [11, 11, 10, 0] (interleaved M-RoPE; = plain NeoX for text) |
| `rope.freq_base` | 1e7 |
| `ssm.conv_kernel` / `state_size` / `group_count` / `time_step_rank` / `inner_size` | 4 / 128 / 16 (k heads) / 48 (v heads) / 6144 |
| `expert_count` / `expert_used_count` | 512 / 10 |
| `expert_feed_forward_length` / `expert_shared_feed_forward_length` | 640 / 640 |
| `expert_weights_scale`, `expert_gating_func` | absent (softmax, no scale) |
| `hyper_connection.count` / `low_rank` | 4 / 320 |
| `embedding_length_per_layer_input` | 160 (PLE row width) |
| `ple.layers` | [1] (0-indexed) |
| `ple.ngram_size` / `heads_per_ngram` / `conv_kernel` | 3 / 8 / 4 |
| `ple.eos_token_id` / `image_token_id` | 248044 (`<\|endoftext\|>`) / 248056 (`<\|image_pad\|>`) |
| `ple.layer_multipliers` | [23703573157769, 20109073645365, 8052911324071] |
| `ple.head_vocab_sizes` | [20000003, 20000023, 20000033, 20000047, 20000059, 20000063, 20000069, 20000077, 20000081, 20000093, 20000107, 20000147, 20000153, 20000159, 20000161, 20000171] |
| `ple.head_offsets` | running sum: [0, 20000003, 40000026, 60000059, 80000106, 100000165, 120000228, 140000297, 160000374, 180000455, 200000548, 220000655, 240000802, 260000955, 280001114, 300001275]; table has 320,001,536 rows (≥ 300001275 + 20000171 = 320,001,446) |

## Tokenizer and chat template

In the first shard's metadata (both files): `tokenizer.ggml.model = "gpt2"` (byte-level BPE),
`tokenizer.ggml.pre = "qwen35"`, 248,320 `tokens` (= `token_embd` rows), 247,587 `merges`,
`token_type`. `add_bos_token = false`; `bos_token_id` = `padding_token_id` = 248044
(`<|endoftext|>`, also the PLE EOS); `eos_token_id` = 248046 (`<|im_end|>`); `<|im_start|>` is
248045; `<think>` is 248068.

`tokenizer.chat_template` is a 9,993-byte Jinja template (ChatML with `<think>`). With no
arguments it adds a system message "Reasoning effort is set to xhigh. …" and opens the assistant
turn with `<think>\n`; `reasoning_effort` (xhigh/medium/low) and `enable_thinking=false` (emits
`<think>\n\n</think>\n\n`) change that.

## The reference forward and its parity

`tang-llm flash-ref <first shard> <ids…> [--top K] [--last N] [--dump DIR]` runs
`src/flash/reference.rs`: every weight dequantised to f32, f32 activations, f64 for the router
softmax and logprobs, a whole sequence per layer. It prints top-K next-token logprobs per position
as JSON lines. `--dump DIR` writes the last position's intermediates as raw little-endian files plus
`index.json` (name, file, dtype, shape):

| Name | Shape | What |
|---|---|---|
| `embd` | f32 [2560] | token embedding row |
| `ple.rows` | u32 [16] | the n-gram table rows hashed for this token |
| `ple.emb`, `ple.gate`, `ple.out` | f32 [2560], [4], [4, 2560] | gathered rows, per-stream gate, residual after the PLE block |
| `LNN.mixer_out` | f32 [2560] | GDN or QSA output, before the HC write |
| `LNN.post_mixer` | f32 [4, 2560] | residual streams after the mixer's HC write |
| `LNN.gdn_state_norm` | f32 [48] | Frobenius norm of each v head's 128×128 state after this token (GDN layers) |
| `LNN.qsa_selected` | u32 [≤ 2051] | cells QSA attended to, ascending (QSA layers) |
| `LNN.router_ids`, `LNN.router_w` | u32 [10], f32 [10] | top-10 experts in rank order, renormalised weights |
| `LNN.moe_out` | f32 [2560] | routed + shared expert output |
| `LNN.post_moe` | f32 [4, 2560] | residual streams after the MoE's HC write |
| `final_x`, `logits` | f32 [2560], [248320] | output of the final HC read, logits |

Measured on mew (i9-12900K), ISTA Q2_0, against llama.cpp b11382 (`11fe02151`) `llama-server`
CPU-only (`-ngl 0`, no `--gpus`, so the CUDA backend doesn't load; default f16 KV cache,
flash attention auto). llama.cpp's numbers are per prefix with `n_predict: 1`, `n_probs: 10`,
`cache_prompt: true` (one decode per position; `scripts/flash_llama_ref.py probs`), i.e. the
softmax of its raw logits. KL is KL(llama ‖ tang) on {llama's top-10, rest}
(`flash_llama_ref.py compare`):

| Prompt | Positions | Top-1 agree | KL mean | KL median | KL p99 |
|---|---|---|---|---|---|
| chat (template + one question) | 64 | 63/64 (98.4%) | 0.0046 | 0.0003 | 0.094 |
| code (`sample.rs`, 40 lines) | 318 | 314/318 (98.7%) | 0.0035 | 0.00006 | 0.051 |
| long (`engine.rs`, first 2300 tokens) | 2300 | 2241/2300 (97.4%) | 0.0070 | 0.0004 | 0.083 |
| long, positions 2051..2299 (selection active) | 249 | 247/249 (99.2%) | 0.0056 | 0.00003 | 0.041 |

Every top-1 miss checked was a near tie in both (e.g. −0.866 vs −0.892 in llama.cpp, −0.868 vs
−0.831 in tang).

**The remaining distance is the model's sensitivity, not a bug.** `--llama-numerics` re-runs the
reference with ggml-cpu's activation rounding (Q8_0 / Q8_K / BF16 per weight type, f16 QSA cache).
Our own f32 run against our own emulated run differs by as much as either does from llama.cpp:
KL 0.0047 / 0.0024 / 0.0083 and top-1 96.9% / 98.4% / 96.7% on chat / code / long. So any
implementation with int8 activations sits about KL 0.003–0.008 and 97–98% top-1 from f32;
fast kernels should be held to that band against this reference, not to zero.

At 2300 tokens QSA drops at most 63 of 575 blocks, and that changes the output by only
KL 0.0005 (`--qsa-dense` ablation), well under the noise above, so those runs don't by themselves
verify the selection. At 8151 tokens (all of `engine.rs`; last 51 positions; QSA keeps 512 of 2037
complete blocks plus the 3 tail cells, i.e. 2051 cells) selection matters more, and llama.cpp
sides with it:

| 8151 tokens, positions 8100..8150 | Top-1 | KL mean | KL max |
|---|---|---|---|
| llama.cpp vs tang (selection) | 51/51 | 0.0015 | 0.022 |
| llama.cpp vs tang `--qsa-dense` | 51/51 | 0.0036 | 0.118 |
| tang selection vs tang dense | 51/51 | 0.0053 | 0.213 |

At the two positions where selection moves the output most (8149: KL 0.21 selected vs dense; 8105:
0.032), llama.cpp is 5–7× closer to the selected run (0.022 vs 0.118; 0.003 vs 0.022). Weak but
consistent evidence; a direct check would compare `LNN.qsa_selected` with llama.cpp's
`indexer_top_k` tensor through an eval callback, which the server image can't do.

The unsloth file runs too (IQ2_XS/IQ3_XXS/IQ4_NL experts, Q8_0 HC): against llama.cpp on the same
file, chat 61/64 top-1, KL 0.030; code 310/318, KL 0.011. Its own f32-vs-emulated floor is higher
than ISTA's (KL 0.023 / 0.014, top-1 96.9% / 97.5%), and with `--llama-numerics` the distance to
llama.cpp drops to 0.017 / 0.0088, so this too is rounding sensitivity, of a lower-bit quant.

Cost on mew: 10 tokens 8 s (24 threads); 318 tokens 81 s (8 threads, sharing the CPU with llama.cpp);
2300 tokens 502 s (8 threads, shared); 8151 tokens 1715 s (8 threads, shared). Anonymous memory
stays under ~6 GB (2 GB through the layers, the head adds 2.5 GB); the rest of the RSS (up to
38 GB at 8K) is the mapped GGUF in the page cache.

To reproduce (`S=crates/tang-llm/scripts/flash_llama_ref.py`, a CPU llama-server on :18080):

```sh
python3 $S tokenize --chat "What is the capital of France? Answer in one word." > chat.ids
python3 $S probs --ids chat.ids > llama_chat.jsonl
tang-llm flash-ref <first shard> --ids-file chat.ids --top 40 --dump dump_chat > tang_chat.jsonl
python3 $S compare llama_chat.jsonl tang_chat.jsonl [--from I --to J]
```

## Dense requantization for the fast kernels

The GPU kernels take a dense weight as bf16, native Q2_0, or Q4X (tang-Q4: affine 4-bit, group 64,
bf16 scale and bias, 0.5625 B/weight) against int8 activations. ISTA's other dense tensors (3.31 G
weights in nine types) must be requantized or get their own kernels. Measured with
`flash-ref --dense-as <policy> --act-int8` (`src/flash/requant.rs`) against the f32 reference (and
llama.cpp), and `flash-requant` for bytes. "Bytes" is what a decode step reads in full: dense +
HC + router + head, not experts or table rows.

**The int8 activation contract alone** (per 32, `d = amax/127` in f32, round half away from zero,
on every GEMV whose weight isn't kept bf16/f32, experts included; `--act-int8`, weights exact) is
the parity band for the fast engine: KL vs the f32 reference 0.0027 / 0.0016 / 0.0056, top-1
98.4% / 99.1% / 97.6% (chat / code / long), and 0.0016 / 99.0% on the 387-token chat2 sequence.

| Policy (all with int8 activations) | Bytes/step | KL vs f32: chat / code / long | top-1 vs f32: long | KL vs llama.cpp: long |
|---|---|---|---|---|
| native kernels for every type (= the band above) | 3.366 GB | 0.0027 / 0.0016 / 0.0056 | 97.6% | 0.0080 |
| **native Q3_K, IQ4_XS, Q4_K; Q8_0 for the rest (head included)** | **3.698 GB** | **0.0033 / 0.0026 / 0.0064** | **97.4%** | 0.0085 |
| native for all five K-quants; Q8_0 for the 4 small types | 3.389 GB | not run: native is exact, Q8_0 adds ≈ 0 | | |
| Q8_0 for everything | 5.077 GB | — / 0.0020 / — (chat2 0.0015) | | |
| Q4X min/max for everything (the engine today) | 3.422 GB | 0.095 / 0.059 / 0.106 | 88.9% | 0.107 |
| Q4X searched (scale, bias) for everything | 3.422 GB | 0.035 / 0.035 / 0.087 | 90.6% | 0.088 |
| searched Q4X, but Q8_0 for ≥ 5-bit types | 3.859 GB | 0.019 / 0.013 / 0.038 | 93.9% | 0.038 |

Q4X costs real quality: KL 0.06–0.11 and 89–95% top-1, 10–20× the int8 band. Per source type,
searched Q4X on that type alone (rest native, int8 activations; KL vs f32 on code / chat2, band
0.0016 / 0.0016):

| GGUF type | Weights | Native bytes | Q8_0 bytes | Q4X bytes | Q4X cost (KL code / chat2) | Recommendation |
|---|---|---|---|---|---|---|
| `output.weight` (Q5_K) | 636 M | 437.0 MB | 675.4 MB | 357.6 MB | **0.024 / 0.042** | never Q4X; Q8_0 now, native Q5_K later |
| Q6_K (12 tensors) | 103 M | 84.4 MB | 109.3 MB | 57.9 MB | **0.016 / 0.018** | Q8_0 (or native) |
| IQ4_XS (56) | 824 M | 437.8 MB | 875.6 MB | 463.6 MB | 0.008 / 0.009 | native (Q4X is *more* bytes) |
| Q4_K (38) | 413 M | 232.2 MB | 438.7 MB | 232.2 MB | 0.007 / 0.009 | native (same bytes as Q4X) |
| Q3_K (90) | 1,162 M | 499.3 MB | 1,234.6 MB | 653.6 MB | 0.007 / 0.006 | native (Q4X is 31% *more* bytes) |
| Q5_K except the head (15) | 122 M | 83.6 MB | 129.2 MB | 68.4 MB | 0.0044 / 0.0038 | Q8_0 (or native) |
| IQ4_NL, Q4_0, Q5_0, Q8_0 (32 shared-expert downs) | 52 M | 32.6 MB | 55.7 MB | 29.5 MB | 0.0034 / 0.0041 | Q8_0 |

So the cheapest fix is not better Q4X rounding (searched scales halve the cost and it's still
0.035–0.09): it's three native GEMVs (Q3_K, IQ4_XS, Q4_K: ggml's `vec_dot_*_q8_1` MMVQ paths, with
the same int8 activations) plus an int8 Q8_0 GEMV for everything else, at 3.70 GB/step (+10% over
the file's 3.37 GB, +8% over Q4X's 3.42 GB) and parity at the int8 band. Native Q5_K and Q6_K
kernels too would bring it to 3.39 GB. Weight-space error, for reference (relative ‖W−Q(W)‖²,
`flash-requant`): Q4X 0.85–1.17%, searched Q4X 0.54–0.97%, Q8_0 ≤ 0.004%, for every source type.

## MTP acceptance

`tang-llm flash-mtp <main> <mtp> --ids-file F --from I --depth 3` (`src/flash/mtp.rs`):
teacher-forced chains of depth 1..3 from every start, compared with the main model's own greedy
token at each step (a depth-`k` draft is counted only when the drafts before it were accepted and
the main model's greedy tokens are the sequence's, which holds on its own greedy text: the
`*_gen.ids` sequences are a prompt plus llama.cpp's greedy continuation). ISTA main model, the
Q8_0 MTP file with its own embedding and head (the main model's head gives the same: 0.761 /
0.675 / 0.689 on chat2), f32, dense MTP attention:

| Sequence (scored region) | Starts | Depth 1 | Depth 2 (conditional) | Depth 3 (conditional) | Cumulative 1 / 2 / 3 |
|---|---|---|---|---|---|
| code continuation (128 greedy tokens after `sample.rs`) | 126 | 0.897 | 0.882 | 0.851 | 0.897 / 0.791 / 0.673 |
| code, whole sequence (prompt + continuation) | 444 | 0.919 | 0.901 | 0.869 | 0.919 / 0.828 / 0.719 |
| chat2 reasoning (320 greedy tokens, a `<think>` answer) | 318 | 0.761 | 0.671 | 0.662 | 0.761 / 0.511 / 0.338 |
| chat (94 tokens: one-word answer, 30 greedy tokens) | 28 | 0.857 | 0.913 | 0.900 | 0.857 / 0.783 / 0.704 |
| long (`engine.rs` 2300 tokens, teacher-forced prompt) | 2298 | 0.828 | 0.831 | 0.868 | 0.828 / 0.688 / 0.597 |

That's the published band for this family (≈ 0.9 / 0.72–0.78 / 0.48–0.62 cumulative) on code
and below it on free-form reasoning text. With two drafts always proposed, chat2's numbers give
(0.761 + 0.511) / 2 = 0.64 accepted per drafted token, against unsloth's "draft acceptance
0.661" for llama.cpp `--spec-draft-n-max 2` on its own runs.

Calibration (all depths; acceptance by the draft's own softmax probability): on code
0.98 at p ≥ 0.9, 0.60–0.92 in [0.5, 0.9), 0.33–0.57 below 0.5; on chat2 0.995 at p ≥ 0.9, 0.71–0.86
in [0.5, 0.9), 0.29–0.62 below. Gating at p ≥ 0.5 keeps 93.5% (code) / 88.3% (chat2) acceptance on
the drafts it lets through and rejects drafts that would have been accepted 45% / 44% of the
time.

Checked by ablation on chat2 depth 1: per-stream `hnorm` 0.761 vs Strata's RMS over all 10240,
0.711; adding 1 to `enorm`/`hnorm` (as if stored raw): 0.610; `[h ; e]` instead of `[e ; h]`: 0;
rope offset by one position: unchanged (only relative positions matter). llama.cpp b11382's own
`--spec-type draft-mtp` crashes on the CPU backend (`llama_kv_cache::set_input_k_idxs` abort in
`common_context_can_seq_rm`), so there is no llama.cpp number to compare.

`flash-mtp --dump DIR` adds `mtp.x_attn.<pos>`, `mtp.r_out.<pos>`, `mtp.final_x.<pos>` (the last
teacher-forced cell's attention input, output residual and head input) and `mtp.tf_top`,
`mtp.tf_top_p`, `mtp.tf_tokens_in` to the main model's dump. On mew: `~/flash-truth/dump_mtp_code`.

## QSA selection, checked directly

`scripts/flash_qsa_topk.cpp` (a replacement for llama.cpp's `eval-callback`, built CPU-only from
llama.cpp `46847e6` on mew) dumps `indexer_top_k` and `indexer_score` for the last token of the
8151-token prompt; `scripts/flash_qsa_compare.py` diffs them with `LNN.qsa_selected` and
`LNN.qsa_scores`.

| Layer | 3 | 7 | 11 | 15 | 19 | 23 | 27 | 31 | 35 | 39 | 43 | 47 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| same blocks (of 512) | 509 | 506 | 508 | 507 | 506 | 506 | 504 | 495 | 487 | 480 | 477 | 482 |
| score corr. (tang vs llama.cpp) | 0.9998 | 0.9999 | 0.9999 | 0.9998 | 0.9996 | 0.9989 | 0.9987 | 0.9957 | 0.9922 | 0.9887 | 0.9878 | 0.9917 |

Selection logic agrees: in layers 3–15 every disagreeing block sits within ~50 places of the cut
in llama.cpp's own ranking (ranks 491–566 around 512), i.e. near-ties at the 512th place. Deeper, the scores themselves drift, a few
blocks a lot (block 359: 6.58 vs 3.17 at layer 39). That drift is numeric, not a bug: our own f32
run against our own `--llama-numerics` run moves the same blocks (673, 1033, 468, 78, 359, 1096,
1814) by similar amounts and its overlap falls the same way (509 at layer 3, 495 at layer 27). The
indexer's ranking near the cut is simply that sensitive; it costs nothing visible at the output
(KL 0.0015 vs llama.cpp over the last 51 positions).

## The MTP file

`~/models/qwen3.8-flash-next/MTP/mtp-Qwen3.8-Flash-Next-Q8_0.gguf` (unsloth, 4,137,429,120 bytes,
one shard, "self-contained": it carries its own embedding and head). Architecture `qwen4exp` with
`block_count = 49` and `nextn_predict_layers = 1`: the MTP block is `blk.48`, and the file has no
trunk. `attention.compress_ratios` has 49 entries and is **0 at index 48**, so the MTP attention is
dense; the block still ships indexer tensors, which llama.cpp loads and doesn't use. Its other
metadata matches the main files (no `ple.*` keys).

| Pattern | ne | Type | Bytes | Role |
|---|---|---|---|---|
| `token_embd.weight` | `[2560, 248320]` | Q8_0 | 675,430,400 | embedding (same model as the main file's) |
| `output.weight` | `[2560, 248320]` | Q8_0 | 675,430,400 | LM head (same) |
| `blk.48.nextn.enorm.weight` | `[2560]` | F32 | 10,240 | RMSNorm gamma on the token embedding, already 1+w |
| `blk.48.nextn.hnorm.weight` | `[10240]` | F32 | 40,960 | per-stream RMSNorm gamma on the main residual, read `[2560, 4]`, already 1+w |
| `blk.48.nextn.eh_proj.weight` | `[5120, 2560]` | Q8_0 | 13,926,400 | `[e ; hn[c]] → R[c]`, per stream (embedding half first) |
| `blk.48.hc_attn_{norm,down,up,inject}.weight` | as main | F32 / Q8_0 | 7,047,680 | HC before attention |
| `blk.48.attn_{q,k,v,output}.weight` | as main QSA | Q8_0 | 52,920,320 | attention (q is `[q 256 \| gate 256]` × 24) |
| `blk.48.attn_{q,k}_norm.weight` | `[256]` | F32 | 2,048 | q/k RMSNorm |
| `blk.48.indexer.{q_proj,k_proj}.weight`, `indexer.{q,k}_norm.weight` | as main | BF16 / F32 | 3,277,824 | unused (compress ratio 0) |
| `blk.48.hc_ffn_{norm,down,up,inject}.weight` | as main | F32 / Q8_0 | 7,047,680 | HC before the MoE |
| `blk.48.ffn_gate_inp.weight`, `ffn_gate_inp_shexp.weight` | `[2560, 512]`, `[2560]` | F32 | 5,253,120 | router, shared gate |
| `blk.48.ffn_{gate,up,down}_exps.weight` | `[2560, 640, 512]` / `[640, 2560, 512]` | Q8_0 | 2,673,868,800 | 512 routed experts (5,222,400 B each) |
| `blk.48.ffn_{gate,up,down}_shexp.weight` | `[2560, 640]` / `[640, 2560]` | Q8_0 | 5,222,400 | shared expert |
| `blk.48.nextn.hc_head_{norm,down,up}.weight` | `[10240]`, `[10240, 320]`, `[320, 10240]` | F32 / Q8_0 | 7,004,160 | the MTP's own final HC read before the head |

Totals: experts 2.674 GB, embedding 0.675 GB, head 0.675 GB, the rest 0.102 GB; 4.126 GB.

The math (`src/flash/mtp.rs`; llama.cpp `graph_mtp` in `qwen4exp.cpp`): cell `i` takes the main
model's final residual `h_i` (all 4 streams, before `output_hc_*`) and the token at `i + 1`, at
rope position `i`, and predicts the token at `i + 2`:

```
e     = rmsnorm(embed(tok)) * enorm
hn[c] = rmsnorm(h[c]) * hnorm[c]                 per 2560 stream
R[c]  = eh_proj @ [e ; hn[c]]
x, inj = hc_read(R, hc_attn_*);  R = hc_write(R, attn(x), inj)   dense causal, own K/V cache
x, inj = hc_read(R, hc_ffn_*);   R = hc_write(R, moe(x), inj)
logits = output @ hc_read(R, nextn.hc_head_*)
```

`R` is the next chained draft's `h`: depth `d` from start `i` is a cell at position `i + d − 1`
whose token is the previous draft, attending to the teacher-forced cells `0..=i` and the chain's
own earlier cells.

## Corrections

What the GGUF and llama.cpp (`46847e6`, 2026-10-04; the docker image is build 11382 `11fe02151`,
same day) show, where [strata.md](strata.md) (written from Strata, which pins llama.cpp
`3cf03257`) says otherwise or is silent:

1. **QSA selection is by block, not by cell.** Current llama.cpp scores each *complete* 4-cell
   block, keeps the top `min(n_blocks, top_k / 4 = 512)` blocks, and always adds the incomplete
   tail's 0–3 cells (the current token included). strata.md's "pick top 2051 cells (blocks
   weighted by cell count)" with a +1e9 tail bias picks the same cells only when the tail is
   non-empty; when `n_kv % 4 == 0` cell-level top-2051 adds 3 cells of a 513th block that
   llama.cpp doesn't. Selection stops being the identity once a token sees more than 512 complete
   blocks, i.e. from position 2051 (`n_kv ≥ 2052`), not 2048.
2. **Indexer score has a 1/√128 head weight:** `score[b] = Σ_h relu(q_h · k_b) / √idx_dim`
   (`ggml_lightning_indexer` with prescaled weights). It doesn't change the ranking, but a kernel
   reproducing scores (or comparing dumps) needs it. strata.md: "no softmax, no scale".
3. **Which weights are BF16 depends on the file.** In ISTA the HC matrices, `ple_value`, the
   router, `ffn_gate_inp_shexp`, `ssm_alpha/beta` and the indexer projections are BF16; in unsloth
   the HC matrices and `ple_key/value` are Q8_0 and the router, `ssm_alpha/beta`, the shared gate
   and `hc_*_inject` are F32. strata.md's "Hyper-connection weights 1.3 GB bf16" is ISTA-only.
4. **ISTA dense types vary per layer** (see the table): ten types across the dense tensors,
   including Q2_0 in some `attn_qkv`, `attn_gate`, `attn_q`, shared-expert and `ple_key` tensors.
   Kernels need the per-tensor type, not a per-role one.
5. **PLE gate clamps:** `gate = sigmoid(sign(s) · √max(|s|, 1e-6))`, `sign(0) = 0`. strata.md
   writes `√|s|`.
6. **RoPE is interleaved M-RoPE** (`LLAMA_ROPE_TYPE_IMROPE`, sections [11, 11, 10, 0], 32 pairs =
   `n_rot / 2`). For text every section gets the same position (and pooled indexer keys are
   rotated with all four set to the block's first position), so it reduces exactly to NeoX on the
   first 64 dims, pairs `(i, i + 32)`, θ base 1e7 — which is what strata.md's "rope64 neox"
   means. Images would differ.
7. **The PLE predecessor window** comes from the sequence's own earlier tokens; before the start,
   a missing predecessor reads as EOS (248044), and an EOS predecessor cuts everything older.
   strata.md's "hash(tok, prev1, prev2)" is right but silent on this.
8. **MTP `hnorm` is per stream.** llama.cpp's `graph_mtp` normalizes each 2560 stream of the
   main residual; Strata's `mtp.cpp` takes one RMS over all 10240. Per stream accepts more
   (0.761 vs 0.711 at depth 1 on chat2). The MTP attention is dense (`compress_ratios[48] = 0`).
9. **No MTP layer in either GGUF.** strata.md's byte table lists "MTP layer 0.8 GB"; that's from
   Strata's separate pack, not these files.
10. **Byte totals, measured (ISTA):** routed experts 33.97 GB (31.64 GiB), matching "34 GB";
   n-gram table 28.80 GB ✓; head 0.437 GB plus 13 MB of `output_hc_*` ✓; HC 1.27 GB ✓ "1.3 GB".
   "Mixers, routers, shared experts 1.8 GB" is 1.65 GB here (PLE block included, embedding not);
   with the 0.27 GB embedding it's 1.92 GB.

Everything else in strata.md's block math matched what the reference needed to agree with
llama.cpp: modulo GDN head pairing, eps on the squared L2 norm, decay before the delta update and
readout from the updated state, `ssm_a` already holding `-exp(A_log)`, sigmoid (not SiLU) GDN and
QSA output gates, pool-then-norm-then-rope indexer keys rotated at the block's first cell, KV head
= q head / 12, `silu(down·xn / 4)` inside the HC read and `2σ(inj / 4)` in the write, 1+w HC
norms, PLE before layer 1's mixer read, router softmax over all 512 then top-10 renormalised with
the 2⁻¹⁴ clamp, and no final norm beyond `output_hc_*`.
