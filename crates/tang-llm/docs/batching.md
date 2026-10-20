# Batching

Several requests decode together on one GPU. Before this, the worker thread took one job from
[`queue.rs`](../src/queue.rs) and ran it to completion with `Engine::complete_with`; every decode
step read every weight to produce one token. Decode is memory-bound, so reading the weights once
for B streams costs about what it costs for one: per-stream speed falls a little as the batch
grows, and total throughput rises nearly B-fold until compute catches up.

## Shape

### Streams

`complete_with` becomes three pieces with the same behaviour:

- `Engine::admit(req, prompt) -> Stream`: render and check the prompt, pick the slot
  ([`slots.rs`](../src/slots.rs)), attach shared blocks, restore from disk, and take the slot's
  cache *out* of the engine into the stream. The stream owns its `Cache`, sampler, parser,
  think budget, grammar, draft session, output tokens and detokenizer state.
- `Engine::step(&mut [Lane], budget)`: one forward pass for every stream in the batch (a
  `Lane` is a stream with its piece sink and `Control`), then sampling and delivery per
  stream. A stream is prefilling (its next chunk of prompt), decoding (its last token, plus a
  draft window when speculating), done, or yielded.
- `Engine::retire(stream) -> Outcome`: seal its blocks and put its cache back as the active
  slot (the previous active one is parked), so the next request in that conversation finds it
  and `save` writes it to disk.

`complete_with` is `admit`, then `step` on a batch of one until done, then `retire`. Solo and
batched runs share the code path, which is what keeps batched output identical to solo output.

### Slots and the KV budget

The engine keeps an active cache plus parked ones, `slots` in all. A running stream checks its
slot's cache out; parked caches stay where they are. The rules:

- At most `slots` streams run (each needs its own cache).
- Two requests with the same `prompt_cache_key` never run together (they'd share one cache);
  the later one waits in the queue.
- A stream joins a running batch only if the pool has room for its prompt plus a typical reply
  (`Engine::kv_room`; `make_room` evicts least recently used *parked* caches, and a running
  stream's blocks are held and never evicted). Past that the pool grows beyond its budget, as it
  always has for one long conversation.

### Forward over segments

`Model::forward_batch(&mut [Segment])`, where a segment is (tokens, cache, whether to return
every row's logits or just the last). The tokens of all segments are concatenated into
`[S, hidden]`, one-token segments first. Per layer:

- norms, the QKV projection, the output projection and the MLP run once over the short
  segments' rows (decode and verify windows, up to 8 tokens each): a matrix-vector product
  becomes a skinny matmul that reads the weights once for all of them. On CUDA, 4-bit weights
  at 1–8 rows go through `gemv_q4_xs4_m{1..8}`, below. A prefill chunk gets its own matmul at
  the size it would have alone.
- the one-token segments (plain decode) go through `ComputeDevice::attention_paged_rows`: QK
  norm, RoPE, the KV write and causal paged attention for every row at its own position over its
  own block table. On CUDA that's three launches per layer for the whole batch
  (`attention_prep_rows`, `attn_partial_rows`, `attn_combine`); each row's keys are split as its
  solo decode would split them, so the result is bitwise the solo one. The default (Metal, CPU)
  runs the per-row ops in a loop.
- longer segments (prefill chunks, speculative verify windows) run `attention_prep_paged` /
  `kv_attention_paged` on their slice of `qkv`, as before.

`forward` keeps its single-sequence path; `Engine` only calls `forward_batch`.

### Speculation

Each stream drafts on its own (a `Drafter`: the suffix-automaton session without the borrow, so
it lives in the stream; or the draft model when that has nothing) and feeds `fed + draft` as its
segment, asking for every row. Acceptance is per stream, exactly as solo: each token is sampled
from the target's logits in order with the stream's own sampler, so output doesn't change. The
cost model learns only from solo forwards (a batched forward's time isn't one stream's). Finished
streams' outputs go into the global automaton once no stream is drafting from it.

### Prefill under a time budget

A prefilling stream's segment is its next chunk. While anything decodes, a step has a time
budget (`TANG_STEP_MS`, default 50 ms: 20 tokens a second for every decoding stream): decode
rows go in first, then prefilling streams, in lane order, each get the largest chunk the
budget's remainder fits, from a fitted model of step time ([`step_cost.rs`](../src/step_cost.rs)):

```text
secs = a + f·decode_rows + Σ_chunks (g_k + b_k·q + d_k·q·c + e_k·q²)
```

for a chunk of `q` new tokens after `c` cached, with separate coefficients for small chunks
(≤ 32 tokens) and large ones, because the kernels differ: on CUDA a 9–32-token chunk runs
`gemm_cols` and split-KV attention, a larger one dequantizes for cuBLAS and runs tiled prefill
attention, and a 33-token chunk costs twice a 32-token one (`bench_batch --probe`; Qwen3-8B
4-bit on the 3090, step ms with 4 decode rows):

| prefix | q=0 | 4 | 9 | 32 | 33 | 64 | 128 | 256 | 512 |
|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|
| 0 | 12.7 | 18.8 | 41.5 | 65.4 | 148 | 154 | 206 | 305 | 531 |
| 4096 | 13.2 | 21.9 | 46.9 | 77.5 | 256 | 281 | 454 | 788 | 1365 |
| 16000 | 12.8 | 27.6 | 64.9 | 112 | 622 | 664 | 1209 | 2313 | 4066 |

The solver takes the larger of the two regimes' fits (`StepCost::fit_chunk`). A chunk is at
least 9 tokens, past the decode rows' share of the matmuls; the first prefilling stream always
gets one, so prefill moves even when the budget can't hold it. With nothing decoding, a chunk is
a whole `PREFILL_CHUNK` (512), as a prompt gets alone.

The model is calibrated at load (`Engine::calibrate`: decode at 1–8 rows and chunks of both
regimes at two prefix lengths, ~4 s for Qwen3-8B on the 3090 once the kernels are compiled;
`TANG_CALIBRATE=0` skips it) and refined from every step: least squares with older steps decayed,
ridged toward the calibration so a long run of identical steps can't move the terms they don't
constrain. `/node` shows it as `step_cost`.

Images keep the old path: a stream with images prefills in one go inside `admit`.

Decode inter-token latency while a 16k-token prompt prefills next to 4 decoding streams
(Qwen3-8B 4-bit, RTX 3090; decode alone is 13.6 ms a step):

| step budget | step p50 | p99 | max | prefill |
|--|--:|--:|--:|--:|
| none (whole 512-token chunks) | 2198 ms | 3943 | 3943 | 69 s (231 tok/s) |
| 100 ms | 88.5 | 99.7 | 209 | 50 s (318 tok/s) |
| 50 ms | 56.0 | 66.9 | 150 | 93 s (172 tok/s) |

At 50 ms the steps run a little over: past ~8k of prefix even a 9-token chunk costs more than
the 37 ms the decode rows leave, and prefill still has to move. A 100 ms budget prefills faster
than whole chunks do, because 512-token chunks at a long prefix hit the slow tiled attention
(`attn_prefill` with few query tiles over many keys), and 33–256-token chunks hit it less.
That kernel, and the dequantize-for-cuBLAS path for 33+ rows, are what make prefill here slow;
fixing them moves every number in these tables.

The bigger chunks and varying sizes mean a prompt prefilled beside decoding streams isn't
prefilled in the same chunks as alone, so its cache can differ from the solo one in the last
bits (see Exactness). Decode rows are unaffected.

### Scheduler

The rules are in [`batch.rs`](../src/batch.rs), pure and unit-tested:

- `joins`: between steps a waiting request joins while there's a free slot (`max_batch`: one
  per KV slot, capped by `TANG_MAX_BATCH`), its conversation isn't already running, and the KV
  pool has room for its prompt plus a typical reply. An empty batch takes anything. The worker
  takes joiners with `Queue::pop_where`, interactive lane first, skipping ones that can't join.
- `lane_order`: the prefill budget goes to interactive streams first, then by arrival.
- `yields`: a background stream that's still prefilling gives way when interactive work is
  waiting and the batch is full (`Outcome::Yielded`, sealed, back to the front of its lane). A
  decoding one keeps going.
- Loads and unloads wait for the batch to drain.

The worker loop: admit joiners, step, retire what finished (reply, usage log, `after`), repeat;
block on the queue only when the batch is empty. A backend that doesn't batch (Flash) keeps the
serial path. `/node`'s `queue.running` lists every running request. The HTTP API doesn't change.

## Steps

1. `Stream` / `admit` / `step` / `retire` with `complete_with` on top; the `batch.rs` rules;
   the worker runs batches. The existing GPU tests pass unchanged.
2. `Model::forward_batch`; batched decode; `attention_paged_rows` (CUDA kernels, default
   elsewhere). Test: `model::a_batched_forward_matches_each_segment_alone` (CPU).
3. Batched speculation. `tests/speculate.rs::batched_matches_alone`: the eight requests together,
   drafting off and on, all at once and staggered, byte-identical to each alone.
4. Chunked prefill in decode steps. `tests/batch.rs`: a 3000-word prompt joins two decoding
   streams, which get a token every step while it prefills; a background prefill yields inside a
   batch and goes on later; the HTTP server decodes four concurrent requests together (`/node`
   shows them) with the same replies as one at a time.
5. `examples/bench_batch.rs`, below.

## Exactness

A row's numbers must not depend on the other rows. Embedding, norms, activations and attention
run per row (the fused decode attention splits each row's keys as its solo decode does). For
the matmuls:

- `gemv_q4_xs4_m{M}` (CUDA, 4-bit, 1–8 rows) sums each row in the same order for every M, so a
  decode row in a batch of up to 8 streams gets the bits it gets alone.
- Prefill chunks get their own matmul, and the head runs in pieces of at most 8 rows. With
  nothing decoding a chunk is the solo size; beside decoding streams the time budget sizes it.

So on CUDA with 4-bit weights, plain batched decode matches solo byte for byte, bf16 KV
included (the bench's 8 streams × 256 tokens: 8/8). A prompt prefilled beside decoding streams
gets budget-sized chunks, so its own cache, and so its reply, can differ in the last bits. What still isn't batch-invariant: verify
windows when the batch's short rows total more than 8 (`gemm_cols16/32`), bf16 weights
(`gemv_bf16` vs `gemm_cols`), and Metal's kernels. With bf16 KV those differences can flip a near
tie late in a long reply, the same effect `tests/speculate.rs` documents for verify windows;
with f32 KV (the tests) every output matched on CUDA and Metal.

## Results

`bench_batch`: greedy, essays, 128 tokens per stream after prefill; per-stream is a stream's
tokens over the decode time, total is the batch's.

Qwen3-8B (mlx-community 4-bit) on the RTX 3090 (`mew`), bf16 KV, 256 tokens a stream:

| batch | per-stream tok/s | total tok/s | ms/step | same as alone |
|------:|-----------------:|------------:|--------:|--------------:|
| 1 | 108.3 | 108.3 | 9.2 | 1/1 |
| 2 | 92.3 | 184.6 | 10.8 | 2/2 |
| 4 | 71.4 | 285.5 | 14.0 | 4/4 |
| 8 | 49.4 | 395.0 | 20.3 | 8/8 |

How it got there (total tok/s at batch 1 / 2 / 4 / 8, 128-token runs):

| | 1 | 2 | 4 | 8 |
|--|--:|--:|--:|--:|
| per-stream attention, `gemv_q4` / `gemm_cols8` | 95 | 141 | 179 | 288 |
| + fused decode attention | 98 | 155 | 211 | 395 |
| + `gemv_q4_xs4` | 113 | 191 | 295 | 405 |

`gemv_q4_xs4` (one 4-bit Qwen3-8B matmul, µs; `examples/q4_rows`, old kernel in brackets):

| K × N | m=1 | m=2 | m=4 | m=8 |
|--|--:|--:|--:|--:|
| 4096 × 6144 (qkv) | 26.6 [30.9] | 31.5 [37.5] | 44.8 [55.5] | 65.3 [57.2] |
| 4096 × 4096 (o) | 19.6 [22.9] | 23.9 [31.9] | 30.0 [41.1] | 43.5 [43.3] |
| 4096 × 24576 (gate+up) | 77.9 [101.3] | 92.0 [119.2] | 128.7 [218.5] | 196.0 [214.0] |
| 12288 × 4096 (down) | 52.8 [60.3] | 63.5 [101.1] | 78.8 [119.1] | 125.3 [132.1] |

It's a warp per 4 output columns, each word dequantized once for all rows, the rows'
activations staged in shared memory, and the row count a template parameter (no branches in the
row loop). At 8 rows it's compute-bound (~2.5x its 1-row time: 8 FMAs a weight plus decode,
~38% of the 3090's issue rate), so batch 8 barely moves; batches 2–4, where the old kernels paid
for padding to 8 rows, gain the most. Things that didn't help: replacing the int-to-float
convert alone, loading activations straight from L1 (they miss to L2: ~800 MB a call at 8 rows),
and 16-warp blocks. Tensor cores don't buy much on a GA102: with f32 accumulation they run at the
FP32 FMA rate.

Qwen3-0.6B 4-bit on the M4 (Metal, default per-row attention; M4 times drift ±2x): 279 / 403 /
450 / 569 tok/s total at batch 1 / 2 / 4 / 8.

At batch 8 the 4-bit matmuls are still ~2/3 of the step. The next step there is a kernel that
does fewer instructions per weight-row at 8 rows (e.g. two columns a thread with packed
dequant), or bigger batches (16–32) on a GPU with more FP32 throughput per byte.
