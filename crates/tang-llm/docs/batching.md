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
- `Engine::step(&mut [&mut Stream], budget, ...)`: one forward pass for every stream in the
  batch, then sampling and delivery per stream. A stream is in one of three phases: prefilling
  (its next chunk of prompt), decoding (its last token, plus a draft window when speculating),
  or done.
- `Engine::retire(stream) -> Outcome`: seal its blocks and put its cache back among the
  parked slots, so the next request in that conversation finds it.

`complete_with` is `admit`, then `step` on a batch of one until done, then `retire`. Solo and
batched runs share the code path, which is what keeps batched output identical to solo output.

### Slots and the KV budget

The engine keeps an active cache plus parked ones, `slots` in all. A running stream checks its
slot's cache out; parked caches stay where they are. The rules:

- At most `slots` streams run (each needs its own cache).
- Two requests with the same `prompt_cache_key` never run together (they'd share one cache);
  the later one waits in the queue.
- A stream is admitted only if the pool has room for its prompt plus a typical reply
  (`make_room` evicts least recently used *parked* caches; a running stream's blocks are held
  and never evicted). A stream that runs the pool dry mid-decode finishes with `Length`, like
  a full context.

### Forward over segments

`Model::forward_batch(&mut [Segment])`, where a segment is (tokens, cache, which logits rows
to return). The tokens of all segments are concatenated into `[S, hidden]`. Per layer:

- norms, the QKV projection, the output projection and the MLP run once over all `S` rows: a
  matrix-vector product becomes a skinny matmul. On CUDA, `linear` already picks the GEMV for
  `m <= 8` (weights read once for all rows) and column GEMM / cuBLAS above that; Metal's
  `linear` does the same. No new kernels.
- RoPE, the KV write and attention run per segment on its slice of `qkv`, at its own position
  over its own paged KV (`attention_prep_paged` / `kv_attention_paged` with that cache's block
  table). Their outputs are written back into one `[S, q_dim]` buffer on the device.

`forward` is `forward_batch` with one segment. The per-segment attention costs a few small
launches per stream per layer; a varlen paged attention kernel taking every segment at once is
the follow-up if those launches show up at batch 8.

### Speculation

Each stream drafts on its own (suffix automaton session, or the draft model when that has
nothing) and feeds `fed + draft` as its segment, asking for every row. Acceptance is per
stream, exactly as solo: each token is sampled from the target's logits in order with the
stream's own sampler, so output doesn't change. Streams with no draft feed one token; with
mixed widths the batch's rows are the sum. The draft model runs per stream (it's small).
The cost model's widths are per stream as before; a batched forward is timed against its total
rows.

### Prefill under a token budget

A prefilling stream's segment is its next chunk. Each step has a token budget (default 512, the
old `PREFILL_CHUNK`): decoding streams' rows go in first, then prefill chunks fill what's left,
interactive streams before background ones. So a long prompt arriving doesn't stall the
streams already decoding for a whole prefill, and decode steps stay short.

Images keep the old path: a stream with images prefills alone, in one go, before it joins.

### Scheduler

`Batcher` (in [`batch.rs`](../src/batch.rs)) decides who's in the batch between steps; it's pure
(tickets in, decisions out) so it's tested without a model.

- Between steps it admits from the queue while there's a free slot, the conversation isn't
  already running, and the KV budget has room: interactive lane first.
- Background prefill still yields: when interactive work is waiting and there's no slot for it,
  a background stream that's still prefilling is stopped (`Outcome::Yielded`, sealed, pushed
  back to the front of its lane) to free its slot. A background stream that's decoding keeps
  going; it isn't holding up the interactive one past one step.
- Load and unload jobs drain the batch first.

The server's worker loop becomes: pop jobs into the batch, step, deliver, retire finished
streams, repeat; block on the queue only when the batch is empty. `/node` reports every
running request. The HTTP API doesn't change.

## Steps

1. `Stream` / `admit` / `step` / `retire` refactor with `complete_with` on top; `Batcher` with
   unit tests; the worker runs batches. Existing tests (`speculate`, `shared_prefix`,
   `yield_prefill`, `cancel`, `node_api`) pass unchanged.
2. `Model::forward_batch`; batched decode. Test: B streams decoded together give the same
   tokens as each alone (CPU tiny model in-crate; CUDA on mew with a real Qwen3).
3. Batched speculation: per-stream drafts verified in one forward. `tests/speculate.rs` gets a
   batched case: outputs equal solo, drafting on and off.
4. Chunked prefill mixed into decode steps under the per-step token budget. Test: a long prompt
   admitted mid-batch; decoding streams keep producing every step; outputs unchanged.
5. Bench on mew (RTX 3090): per-stream and total tok/s at batch 1/2/4/8 for Qwen3-8B 4-bit
   (`bench_batch` example).

## Exactness

Batched output equals solo output token for token as long as each row's numbers don't depend on
the other rows. That holds for the per-row GEMV (`m <= 8`); the GEMM paths may sum in a
different order for different `m`, which changes logits in the last bits, and greedy decoding can
then pick a different token on a near tie. The tests check exact equality at the batch sizes where
the kernel path matches (and on CPU), and the bench reports agreement.

## Results

See the end of this file once step 5 has run.
