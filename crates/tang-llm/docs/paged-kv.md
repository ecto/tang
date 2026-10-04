# Paged, content-addressed KV blocks

Status: design (implementation follows in steps; see "Steps" at the end for what landed).

## Why

Each KV slot owns a contiguous cache that grows by doubling, and a request reuses only what
*its own* slot already holds. frog's conversations share long prefixes (system prompt, tool
definitions, a forked conversation, subagents started from the same context), and each of them
prefills and stores that prefix again. Blocks shared by content fix both: the second
conversation attaches the first's blocks, skips their prefill, and holds no copy of them.

## Blocks

KV positions live in fixed blocks of `B = 256` positions. A block holds, for each layer, K and V
rows for its positions (bf16, `kv_dim` wide), at the same rows of per-layer pool buffers: block
`b` is rows `b*B .. (b+1)*B` of `K[l]` and `V[l]` for every layer `l`.

Why 256:

- Kernels: the tiled prefill kernel reads 32-row tiles of K/V with one `simdgroup_load`; with
  `B` a multiple of 32, a tile never straddles two blocks. A power of two makes the position →
  row mapping a shift and a mask.
- Bookkeeping stays small: a 32k-token conversation is a 128-entry table, and the hash map,
  refcounts and LRU are per block, not per token.
- Disk: a block of Qwen3-4B is 256 × 144 KiB = 36 MiB, a good unit to write once and read back.
- Granularity costs nothing in reuse: a prompt that shares only part of a block copies those
  rows into a fresh block of its own (below), so sharing is still token-exact. What a larger `B`
  costs is the private tail block each conversation holds (at most `B - 1` positions, 36 MiB
  for Qwen3-4B, against a budget of tens of GB), and coarser eviction.

## Identity

A *full* block is identified by `h_i = H(h_{i-1}, tokens[i*B .. (i+1)*B])` (`h_{-1}` a fixed
seed; FNV-1a 64 over the parent hash and the token ids, as the disk tier already uses). The
chain makes a block's id stand for its whole prefix: equal ids mean equal prefixes, so equal
K/V (images aside: a prompt with images never shares or publishes blocks, as today's disk tier
skips them). Each block also keeps its tokens, so a lookup verifies them instead of trusting
the hash.

Only full blocks are shared. A conversation's last, partial block is private and mutable.

## Pool

`Pool` (one per model) owns the per-layer K/V buffers and per-block metadata:

- `refs`: how many sequences' tables hold the block.
- `hash`: `Some(h)` once *sealed* (immutable, findable); `None` while private.
- `tokens`, `parent`, `last_used` (LRU clock), `on_disk`.
- a free list, and `by_hash: HashMap<u64, block>` over sealed blocks.

Allocation takes a free block; else grows the pool (doubling, within the budget); else evicts
the least recently used sealed block with `refs == 0` (dropped from `by_hash`). If every block
is referenced, the engine drops its least recently used parked conversation (as `make_room`
does today) and retries; only the active conversation alone outgrowing the budget lets the pool
go past it.

Growth reallocates each layer's buffers in turn (copy, then free the old one), so the transient
extra memory is one layer's buffer, not a second pool. Nothing is allocated up front: on
unified memory a buffer the GPU touches is resident, so a budget-sized pool would pin it all.

The budget is the same `kv_budget` (half of RAM with `--kv-slots auto`). Without one (CUDA's
default of one slot), the pool is capped at `slots × context window`, what the slots could
reach before.

## Sequences

`Cache` (one per conversation, as now) becomes a block table: `table: Vec<u32>`, `len`,
`tokens`, and a handle to the pool. Dropping it releases its blocks.

- Writing positions `pos..pos+s` (a forward): blocks past the table's end are allocated; the
  block holding `pos` is made private first if it's sealed or shared (*copy-on-write*: a new
  block, rows `0 .. pos % B` copied on device, the old one released).
- `truncate` only moves `len`; nothing is written until the next forward, which then
  copies-on-write as above. Speculative decoding's rollbacks therefore never touch shared
  blocks.
- `seal` (after each request): every full block in the table that isn't sealed gets its hash.
  If `by_hash` already has that hash (another conversation computed the same prefix), the
  table switches to the existing block and the duplicate is freed.
- `attach(prompt)` (before a request): walk the prompt's hash chain through `by_hash`; the
  sequence takes the longest run of matching blocks (refs + 1), then, if the next sealed block
  of the last match's children shares a few more tokens, copies those rows into a new private
  block. The request reuses `max(own prefix, attached prefix)`.

## Attention through a block table

The kernels get the table (a `u32` buffer) and `shift = log2(B)`; position `j` is row
`table[j >> shift] << shift | (j & (B - 1))`. With `shift == 0` they read contiguously as
before, so non-paged callers (training, the vision tower, tests) are unchanged. Per kernel:

- attention prologue: writes k/v of position `pos + s` at its mapped row.
- decode split-KV (`attn_partial`, `attn_decode`) and the multi-query kernel: map each key.
- tiled prefill (Metal): a 32-key tile starts at a multiple of 32, so it maps once and reads 32
  contiguous rows. CUDA's tiled prefill maps per element.

`ComputeDevice` gets `attention_prep_paged` and `kv_attention_paged`; their defaults gather to
contiguous buffers (CPU, and any shape a backend's kernels don't cover). The table is uploaded
once per forward and shared by all layers.

## Disk tier

Keyed by block hash: `<hash>.kvb` holds a sealed block's tokens and rows (bf16,
position-major), written once (a block already on disk is only touched, for LRU). A
conversation's private tail goes to `<key-hash>.tail` (tokens and rows from the last block
boundary), rewritten each turn. Restoring a prompt walks its hash chain: blocks in the pool are
attached, blocks on disk are read into new sealed blocks, then the tail if its tokens continue
the prompt. The old per-conversation `.kv`/`.json` files are left alone and trimmed by the
budget like the rest.

## Determinism

Attaching another conversation's blocks gives bit-identical results to prefilling them when
the blocks were computed by the same kernels on the same batch boundaries, which holds for a
shared prefix that's a multiple of the prefill chunk (512) and `B`. Otherwise the shared rows
can differ from a fresh prefill by float rounding (as reusing one's own cache already does),
not in meaning.

## HTTP

Unchanged: `prompt_cache_key` still picks the conversation (its sequence), `/v1/prefill` still
warms it, `cached_tokens` reports what was reused, now including blocks attached from other
conversations.

## Steps

1. Paged kernels behind `kv_attention_paged` / `attention_prep_paged` (Metal, CUDA, CPU
   default); tests against the contiguous kernels.
2. `Pool` and the block-table `Cache` in the model; logits identical to the contiguous cache.
3. Engine: attach / seal / eviction within the budget; tests for shared-prefix reuse,
   bit-identical greedy output, and the budget.
4. Disk tier by block hash.
