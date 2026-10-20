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

### Prefill under a token budget

A prefilling stream's segment is its next chunk, always a whole `PREFILL_CHUNK` (512) as when it
runs alone. Each step has a token budget (512): decoding streams' rows count first, then
prefilling streams get chunks in lane order while some budget is left (the first always gets
one, so a step overshoots by at most a chunk). So a long prompt arriving doesn't stall the streams already
decoding for its whole prefill: they get a token every step while it goes in chunk by chunk.

Images keep the old path: a stream with images prefills in one go inside `admit`.

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
- Prefill chunks get their own matmul at their solo size, and the head runs in pieces of at most
  8 rows.

So on CUDA with 4-bit weights, plain batched decode matches solo byte for byte, bf16 KV
included (the bench's 8 streams × 256 tokens: 8/8). What still isn't batch-invariant: verify
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
