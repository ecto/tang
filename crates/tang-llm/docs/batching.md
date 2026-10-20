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

- norms, the QKV projection, the output projection and the MLP run once over all `S` rows: a
  matrix-vector product becomes a skinny matmul. On CUDA, `linear` already picks the 4-bit GEMV
  for up to 3 rows, column GEMM (`gemm_cols8/16/32`) up to 32, and cuBLAS above; the weights are
  read once for all rows. Metal's `linear` likewise. No new matmul kernels.
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

A prefilling stream's segment is its next chunk. Each step has a token budget (512, the old
`PREFILL_CHUNK`): decoding streams' rows go in first, then prefill chunks fill what's left (at
least a block), in lane order. So a long prompt arriving doesn't stall the streams already
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

A row's numbers must not depend on the other rows. Embedding, norms, activations and the
attention path are per row. The matmuls aren't always: the CUDA 4-bit GEMV (1–3 rows) and
`gemm_cols8` (4–8) sum in different orders, so a stream decoded in a batch of 4 sees logits a few
ulps off its solo ones; cuBLAS (prefill chunks, which now may hold several streams' tokens) picks
its algorithm by shape. With f32 KV caches (the tests) every output matched on CUDA and Metal.
With bf16 caches, those ulps can round a key to the neighbouring bf16 value and, over a few hundred
tokens, flip a near tie: the 256-token bench saw 1 of 8 streams diverge (late) at batch 8 on
Metal, and some at batch 2–8 on CUDA before the fused attention landed. This is the same effect
`tests/speculate.rs` documents for verify windows; both outputs are the model's. Making decode
bitwise batch-invariant would take one matmul kernel for 1–8 rows with a fixed summation order.

## Results

`bench_batch`: greedy, essays, 128 tokens per stream after prefill; per-stream is a stream's
tokens over the decode time, total is the batch's.

Qwen3-8B (mlx-community 4-bit) on the RTX 3090 (`mew`), bf16 KV:

| batch | per-stream tok/s | total tok/s | ms/step | same as alone |
|------:|-----------------:|------------:|--------:|--------------:|
| 1 | 97.9 | 97.9 | 10.2 | 1/1 |
| 2 | 77.5 | 155.0 | 12.9 | 2/2 |
| 4 | 52.7 | 210.8 | 19.0 | 4/4 |
| 8 | 49.4 | 395.4 | 20.2 | 8/8 |

Before the fused decode attention, batch 8 was 309 tok/s (25.9 ms a step).

Qwen3-0.6B 4-bit on the M4 (Metal, default per-row attention; M4 times drift ±2x): 279 / 403 /
450 / 569 tok/s total at batch 1 / 2 / 4 / 8.

Where batch 8's 20 ms goes (nsys): `gemm_cols8_q4` is ~15.6 ms of it (~2x the 1-row GEMV for
the same weights, ~220 GB/s against the 3090's 936), the rest is norms, adds and launch gaps. The
batch-4 step costs about what batch 8 does because both use `gemm_cols8`. So the next win is the
skinny 4-bit matmul itself (e.g. two columns a thread to halve the shared-memory loads per
weight, or tensor-core MMA with the batch as N=8), not the scheduling.
