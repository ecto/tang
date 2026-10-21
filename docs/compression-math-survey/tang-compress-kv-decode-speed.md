# cram-codec KV codec in the attention loop: decode speed

Branch `claude/kv-decode-speed`. Question: can the KV codec from PR #73 (`kvcodec` GPU layout, 24–28% fewer bits than the best baseline at matched quality) be decoded every generation step without making generation 25–50× slower? Hardware: Apple M4 Max (40-core GPU), Metal, bf16 weights and KV pool. The machine was shared with other GPU apps; every key row was run twice (`*_rep.jsonl`), and the tables use the second run where the two disagree by more than ~10%.

## Verdict

**Not fast enough. The limit is entropy-decode throughput, and it is documented below.**

- The 25–50× slowdown was mostly lane starvation, and that part is fixed. The best in-loop configuration is now 1.8–5.5× a bf16 step.
- What is left is the adaptive binary arithmetic decoder itself. In the model it decodes about 4 Gvalue/s, which is the same ceiling the isolated kernel hits.
- Matching KIVI's step overhead (8–33%) would need about 50 Gvalue/s at Qwen2.5-1.5B 8K, roughly 11× this kernel.
- Making lanes short enough to be fast also costs bits. Lanes of 32 tokens × 4 channels keep about two-thirds of the bit advantage over KIVI-4. Lanes of 16 × 2 keep none of it.

The positioning stands: tkv is for storing and moving caches. Decode it once on load (variant 1), not inside the step.

## Why it was slow

`restore_kv` runs one `tkv_decode` dispatch per layer per K/V, before that layer's attention. One thread decodes one (128-token block, head) substream: 128 × hd values in series.

At 2K on SmolLM2-135M that is 16 blocks × 3 heads = **48 threads per dispatch**, each decoding 8,192 values. Each thread also copied 64 × 147 prior words into its model scratch, with duplicates. The GPU sat almost idle, and the step was 60 serial latency-bound dispatches.

## Variants

All rows use `tkv:0.07` (step = 0.07 × centred rms × √12, K at 0.5×, V at 2×). NLL is over the 24 teacher-forced decode steps after the context. The context text is this repo's markdown, tokenized per model. tkv's NLL matches bf16 to within ±0.03 everywhere, so the bits columns compare at equal quality.

`tkv:rel:B:W` means B-token blocks and W-channel lanes. W = 0 means the whole head; the old layout is `B:0`.

### ms/step and bits/value

| model, ctx | bf16 | KIVI-4 g32 | KIVI-2 g32 | tkv 128:0 (PR #73) | tkv 16:0 (old best) | **tkv 32:4** | **tkv 16:2** | tkv 128 + cache (1) |
|---|---|---|---|---|---|---|---|---|
| SmolLM2-135M 2K | 3.1 | 3.8 | 3.8 | 812 | 120 | 19.4 | **10.4** | 3.1 |
| SmolLM2-135M 8K (8064) | 5.1–5.8 | 7.3 | 7.4 | 817 | 138 | **30.1** | 30.2 | 5.0 |
| Qwen2.5-1.5B 2K | 9.6 | 10.4 | 10.4 | 1577 | 229 | 25.5 | **17.5** | 10.0 |
| Qwen2.5-1.5B 8K | 12.6 | 15.0 | 14.9 | 1570 | 224 | **38.6** | 41.7 | 12.3 |

| model, ctx | KIVI-4 | KIVI-2 | tkv 128:0 | tkv 16:0 | tkv 32:4 | tkv 16:2 |
|---|---|---|---|---|---|---|
| SmolLM2 2K bpv | 5.00 | 3.00 | 3.93 | 3.99 | 4.24 | 5.10 |
| SmolLM2 8K bpv | 5.00 | 3.00 | 3.91 | 3.97 | 4.22 | 5.08 |
| Qwen2.5 2K bpv | 5.00 | 3.00 | 3.85 | 3.90 | 4.18 | 5.04 |
| Qwen2.5 8K bpv | 5.00 | 3.00 | 3.83 | 3.87 | 4.15 | 5.01 |
| NLL Δ vs bf16 (Qwen 8K) | +0.003 | **+0.108** | −0.005 | −0.004 | −0.005 | −0.004 |

KIVI-2 is cheaper in bits but loses quality: NLL +0.05 to +0.25. KIVI-4 is the matched-quality baseline here. Against it, tkv saves 21–23% with 128-token lanes, 15–17% with 32:4, and nothing with 16:2.

Full rows, including 16:4, 128:16 and the option variants: `bench/results/kv3/kvspeed_{smol135,qwen1.5}{,_rep}.jsonl`.

### (1) Cache decoded tiles; decode only newly appended blocks

`KvOpts::cached`, mode suffix `+cache`. Each layer is decoded once, on its first step. Later steps reuse the decoded tiles.

In this prototype new tokens are already raw bf16 after `upto`. Sealing a new 128-token block costs one block encode (CPU) and one block decode, amortised over 128 steps. **ms/step equals bf16** (table above).

The catch is memory. The resident set is the bf16 rows plus the compressed copy, so 16 + 3.9 bits per value instead of 3.9. Nothing is saved while generating. This is the store/move use case, run inside the engine: decompress on load (cache restore, prefix-cache hit, migration between nodes), then run at bf16 speed.

A bounded tile cache doesn't help either. Every step reads every tile once, in the same order. That cyclic pattern defeats LRU, so a cache holding fraction f of the tiles saves at most f of the decode work.

### (2) Parallel / shorter lanes, and their bit cost

`KvParams::lane_ch`, header flag bit 4. A (block, head) substream splits into `hd / W` substreams of W channels. Lane g holds RoPE pairs `g·W/2 ..` (channel c together with c + hd/2), so the kernel still rotates K on write.

Each lane has its own coder state and freshly reset adaptive models. The decoder is bit-identical to the CPU reference at every width (`kvdec_bench`: `bit_identical: true` at W = 0, 16, 4, 2; the unit test covers W = 4 and 2). Streams with W = hd (or `lane_ch = 0`) are byte-identical to the old layout.

Bit cost relative to whole-head 128-token lanes, Qwen2.5 8K:

| lane (tokens × channels) | values/lane | bpv | Δ bits |
|---|---|---|---|
| 128 × 128 | 16,384 | 3.83 | — |
| 16 × 128 | 2,048 | 3.87 | +1% |
| 128 × 16 | 2,048 | 3.86 | +1% |
| 32 × 4 | 128 | 4.15 | +8% |
| 16 × 4 | 64 | 4.45 | +16% |
| 16 × 2 | 32 | 5.01 | +31% |

Below about 128 values per lane, the coder flush (about 4 bytes per substream) and model warm-up dominate.

Speed follows lane count until the GPU is full. At 2K, 16 × 2 is fastest, giving 12K lanes per dispatch for Qwen. At 8K, 32 × 4 already fills the GPU and is as fast as 16 × 2, so it wins on bits.

**Kernel throughput ceiling** (`kvdec_bench`, synthetic AR(1) data, 60 streams in one dispatch, about one SmolLM2 2K step of values):

| lanes | Gvalue/s |
|---|---|
| 128 × 64 (old, 2,880 lanes) | 1.6 |
| 16 × 64 | 3.6 |
| 128 × 16 | 4.4 |
| 16 × 2 | 4.0 |
| 32 × 4 | 2.2–3.1 |

In the model, the decode overhead at 8K is (step − bf16 step) / values per step: 4.4 Gvalue/s for Qwen 32:4 and 3.8 for SmolLM2. **The in-model decode runs at the kernel ceiling.**

Also tried, and removed from the code: a narrow-lane kernel with the adaptive models in threadgroup memory and per-lane state in registers (W ≤ 16; 2–16 lanes per threadgroup to fit 32 KB). It was **slower** at every shape, 0.7–2.5 vs 2.2–4.4 Gvalue/s. Occupancy loss outweighs the cheaper model access. A smaller fix stayed in: the general kernel now copies each dense class's prior once instead of once per channel.

### (3) Fused decode + attention

A fused kernel saves writing the decoded rows and reading them back in attention. To measure that saving without writing the kernel, I used `+nostore` (`KvOpts::decode_only`, lane flag bit 1): the same decode with every output store removed. The decode stays live through the model-scratch writes.

| | store | no store |
|---|---|---|
| SmolLM2 2K, 16:2 | 10.3 | 10.0 |
| SmolLM2 8K, 16:2 | 30.2 | 29.9 |
| Qwen2.5 2K, 16:2 | 18.0 | 17.4 |
| Qwen2.5 8K, 16:2 | 41.7 | 41.4 |

The stores cost 1–3%. A fused kernel can only remove that, plus a little attention read traffic that bf16 already pays. It can't touch the entropy decode, which is the whole gap. Not built.

### (4) Hybrid: recent window raw, older blocks compressed

`KvOpts::window`, suffix `+win<N>`. The newest N positions stay bf16, and only older whole blocks are compressed and decoded each step.

Decode work scales with the compressed share. At 2K with a 1024-token window, Qwen 16:2 goes from 17.5 to 15.6 ms and SmolLM2 from 10.4 to 9.0–9.4 ms. At 8K with a 1024-token window, 7/8 of the cache is still compressed: Qwen 41.7 → 39.8, SmolLM2 30.2 → 27.2.

The memory saving shrinks in proportion, and per-step cost is still linear in old context. A hybrid only helps if the old blocks are decoded less than every step. That needs sparse or retrieval attention, such as page selection by key summaries. Then the codec decodes only the selected pages. That is a different system, listed under follow-ups.

## Needle retrieval

`crates/tang-llm/examples/kvcomp_niah.rs`, same protocol as `bench/kv_hf.py niah`. A "special magic number for <city>" sentence is spliced into a haystack (repo markdown) at depths 0.1, 0.3, 0.5, 0.7 and 0.9. The context is prefilled and compressed. The question and answer are then teacher-forced one decode step at a time through the decoded cache. A case counts as correct if every answer token is the argmax. Qwen2.5-1.5B, 5 depths per context length:

| mode | bpv (2K / 8K) | 2K | 8K |
|---|---|---|---|
| bf16 | 16 | 5/5 | 5/5 |
| KIVI-4 g32 | 5.00 | 5/5 | 5/5 |
| KIVI-2 g32 | 3.00 | 5/5 | 5/5 |
| tkv 128 + cache (variant 1) | 3.86 / 3.79 | 5/5 | 5/5 |
| tkv 16:2 (fastest in-loop) | 5.05 / 4.97 | 5/5 | 5/5 |

Every configuration retrieves every needle, KIVI-2 included, so at these lengths the test doesn't separate them. The tkv variants are error-bounded at the same step whatever the lane shape, so lane shape changes bits, not reconstruction error. Rows: `bench/results/kv3/kvspeed_niah_qwen1.5.jsonl`.

## The limit, stated

A decode step reads every cached value, which is ctx × layers × 2 × kv_dim values: 117 M values for Qwen2.5-1.5B at 8K.

- bf16 attention streams those values at memory bandwidth.
- KIVI dequant is one multiply-add per value, which is close to free.
- tkv is a closed-loop DPCM with adaptive binary arithmetic coding. Each value takes about 4–6 dependent binary decisions, each updating an adaptive model.

On this GPU, that decoder peaks at about 4 Gvalue/s when it has enough lanes. Making lanes short enough to reach that peak at 2K already costs 8–31% bits.

The alternatives don't move this limit:

- Caching decoded tiles gives bf16 speed but no memory saving.
- Fusion saves 1–3%.
- A raw window saves only its share of the cache.

To run inside the step, the entropy stage would have to change. For example: static per-class tables with multi-symbol rANS or tANS decode, the GPU-friendly design of nvCOMP ANS and DietGPU, which decode whole symbols per table lookup instead of bits per model update. That gives up some of the adaptive models' gain, by an amount not measured here.

## Code changes

- `cram-codec::kvcodec`: `KvParams::lane_ch` and `lane_chan()`; GPU substreams per (block, head, channel group); header flag bit 4 plus a lane-width varint (old streams parse unchanged); `GpuTables::lanes` gains the group index.
- `cram-codec::kvgpu`: per-group channel table and dense classes; `LANE_WORDS` 14 → 16 (lane-pair base, full head dim); `Placement::decode_only`.
- `tang-compute` `tkv_decode` (MSL): channel-group lanes (RoPE pairs inside a lane), prior copied once per dense class, decode-only flag, per-lane arrays 256 → 128 (lane width capped at 128).
- `tang-llm::kvcomp`: mode `tkv:rel:block:lane`; `KvOpts { window, cached, decode_only }` and `compress_kv_with`. The bench takes the `+cache`, `+win<N>` and `+nostore` suffixes.
- `tang-llm::model`: **Qwen2 Q/K/V projection biases** (`*_proj.bias`), which were previously ignored. Without them Qwen2.5-1.5B ran with bf16 NLL 15.7 on this text; with them, 2.83. The earlier Qwen3-0.6B numbers are unaffected, since Qwen3 has no QKV bias.
- New example `kvcomp_niah`. `kvdec_bench` gains `--lane`.

## Reproduce

```bash
cargo run --release -p tang-llm --features metal --example kvcomp_bench -- <model-dir> <ids.u32> \
    --ctx 2048,8192 --n 24 --modes bf16,kivi:4:32,tkv:0.07:128,tkv:0.07:32:4,tkv:0.07:16:2,tkv:0.07:16:2+nostore,tkv:0.07:128+cache,tkv:0.07:16:2+win1024
cargo run --release -p tang-llm --features metal --example kvcomp_niah -- <model-dir> <ids.u32> \
    --ctx 2048,8192 --modes bf16,kivi:4:32,tkv:0.07:128+cache,tkv:0.07:16:2
cargo run --release -p tang-compute --features metal --example kvdec_bench -- <x.f32> 3x2048x64 v 0.07 \
    --block 32 --lane 4 --copies 60
```

`<ids.u32>` is little-endian u32 token ids. Here that was all of the repo's `*.md`, concatenated in sorted path order and tokenized with each model's `tokenizer.json`. SmolLM2's "8K" is 8064, because its `max_position_embeddings` is 8192 and the bench needs ctx + n + 8 positions.

## Follow-ups

- Static-table rANS entropy stage for the GPU layout. Measure the bit cost against the adaptive models on real caches, and decode Gvalue/s against the ~50 Gvalue/s needed. This is the only path found that could move the limit.
- Sparse attention over compressed pages (decode only the selected blocks), which would make the hybrid pay off.
