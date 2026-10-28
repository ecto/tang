# GDN memstore: a file store in Gated DeltaNet state

Phase 1 of "zero-context compiled memory". Code: `crates/tang-llm/src/flash/memstore.rs`.

Sources. The idea, the directory/content split, the 1,024-slot hashed directory, the 512-file
limit and the recall numbers are from Divyansh Agrawal's post
[qwenputer](https://divagr.com/blog/qwenputer) and its MIT code
([github.com/divagr18/qwenputer](https://github.com/divagr18/qwenputer), branch `rv-store`:
`docs/q35-store.md`, `chipformer/q35/store.py`, `store_snapshot.py`; checkpoint
`divagr1925/Qwen3.5-4B-Qwenputer-Filesystem` on Hugging Face). No code is ported; where this
differs from the repo's v1 store it says so.

## What phase 1 is

The engine, not the model, drives the store. Every write is a `ComputeDevice::gdn_step` window in
`Commit` mode and every read one in `ReadOnly` mode, at most `MAX_T` = 8 tokens each, whose inputs
are one-hot q/k with g = 1 and β ∈ {1, ≈0}. So the state holds exactly what a compiled read
circuit would later see, and it persists with the recurrent state. The model never reads it yet.

## Layout

Tang's GDN layer state is `[i 128][hv 48][j 128]` f32; v head `hv` uses k head `hv % 16`. The
store owns k heads 12–15 and their 12 v heads (25% of one layer's heads):

| Table | v heads | Rows | Row content |
|---|---|---|---|
| directory | 12–15, 28–31 | 1,024 slots | status, name length, file length, row count, name (96 B, 2 B/float), up to 75 content row numbers |
| content | 44–47 | 512 rows | 127 token ids |

- Slot `s` → v head `12 + s%4 + 16·((s/4)%2)`, row `s/8`. The 8 slots of bucket `s/8` share one
  state row across the 8 directory heads, so a single token with the same one-hot q on the 4
  store k heads reads a whole bucket (plus 4 content rows). Content row `c` → v head `44 + c%4`,
  row `c/4`.
- Name → FNV-1a 64 → home slot; linear probing, deletes leave tombstones (status 2) that a later
  put reuses; a probe stops at an empty slot. The post hashes into 1,024 slots and moves to the
  next slot on collision; the repo's v1 store instead scans a 128-entry directory.
- Column 127 of every written row is a marker 1. The kernel output
  `y = o·rsqrt(mean o² + eps)·norm·σ(z)` loses the row's scale; `row_j = (y_j/norm_j)/(y_127/norm_127)`
  restores it. An unwritten row is all zeros (marker 0).
- A file is a list of content rows held in its slot (no chaining, so a get reads all rows in
  parallel). The repo's v1 store chains 104-token chunks through a meta table instead.

Values are token ids, one per float, below 2^18. The state is exact for integers below 2^24, but a
read goes through the norm and a division (~8 roundings, ~5·10⁻⁷ relative), which needs values
below ~2^20 to round back; 2^18 leaves 4× margin (worst error 0.13) and fits Qwen's 248,320-id
vocabulary, the same 18 id bits the repo's store carries. Names pack 2 bytes per float for the
same reason (3 bytes would reach 2^24).

## Numbers

| | |
|---|---|
| files | 512 (load factor ½ of 1,024 slots) |
| tokens | 512 × 127 = 65,024 (post: ~53k); one file ≤ 75 rows = 9,525 tokens |
| bytes | store heads 12 × 128 × 128 × 4 = 768 KiB; the whole layer state 3 MiB |
| put | 1 probe window (1 token) + ⌈rows-per-k-head / 8⌉ commit windows; a 1-row file: 2 windows, 2–3 tokens |
| get | 1 probe window + ⌈rows/4/8⌉ read windows; a 1-row file: 2 windows, 2 tokens |
| list / open | full directory scan: 128 tokens, 16 windows |

Measured (512 files of 1–127 tokens, 32,891 tokens total): puts 1,075 windows / 1,562 tokens,
gets 1,059 windows / 1,304 tokens, i.e. ~2.1 windows per op. A probe at load ½ almost always ends
in the home bucket (the post's "2–3 steps").

## Driver discipline

A store head is RAM only while every token through it has g = 1 exactly and either a one-hot k
with β = 1 or k = 0; one token with g < 1 scales every row by a non-power of two and the head is
never exact again (pinned in tang-compute's `gdn_free_run_stays_exact`).

Store windows: `a = −40 − dt_bias[h]` on every head (g rounds to 1.0 for any `ssm_a ≥ −10⁴`),
`b = +40` (β = 1.0) on heads written, `b = −40` (β = 4·10⁻¹⁸) with v = 0 elsewhere — the row moves
by `S·4·10⁻¹⁸`, below half an ulp, so it is bitwise unchanged — and q = k = 0 on the 12 base k
heads. Tested: base heads are bitwise untouched by all store traffic.

Model tokens, by construction (weights, not runtime checks). For each store k head `kh`, in the
layer's stacked input projection (`GDN_CONV` + z + a + b rows):

- q rows `kh·128 .. +128`, k rows `16·128 + kh·128 .. +128`: zero, and their conv taps zero → a
  model token has q = k = 0 on store heads, so it cannot write (`S += k·d` with k = 0 is exact)
  and reads 0;
- v rows of the 3 store v heads and their z rows: zero; out-projection columns of those heads
  zero, so the store adds nothing to the residual;
- a rows of store v heads: zero, with `dt_bias = −40` → g = 1 exactly on model tokens too;
- b rows: don't care (k = 0).

The catch is that heads 12–15 are real Coder-Next heads: zeroing them changes the model. Phase 1
uses them to prove the mechanism; the deployable form widens the layer (16→20 k heads, 48→60 v
heads in this layout) or adds the store as an extra GDN layer, so no model head is taken.

`MemStore::audit` reads the state buffer (read-only) and flags any store head with a non-integer,
a value out of `[0, 2^18)`, a marker other than 0/1 or a nonzero unmarked row. One model-like
token (g < 1, dense k) flags all 12 heads in the test. A uniformly decayed row still *reads*
correctly (the marker ratio cancels the scale), which is why the audit, not the read, is the check.

## Persistence

The store is part of the layer's state buffer, so `save_running` / `restore_running` and the
serve disk tier already carry it: `RunningState::parts()[2·layer]` is exactly the buffer
`MemStore` works on. Tested on the raw buffer (the flash `Engine` and `RunningState` are behind
the `cuda` feature): download → upload into a fresh buffer → `MemStore::open` rebuilds the free
list by a directory scan → every file reads back exactly.

Across conversations a full restore would also restore the old chat's base heads, so the store
heads travel alone: `snapshot_heads` (768 KiB) and `restore_heads` splice them into a new
conversation's state, like the repo's `store_snapshot.restore` fills only the persistent heads
of a fresh cache. Tested, including that the new conversation's base heads are untouched.

Wiring still needed: a `memstore` field in the serve session (layer index + snapshot), written to
the disk tier next to the prefix states, and restored into slot 0 at session start.

## How it plugs into frog's memory

The post reports facts that changed between chats recalled 90% of the time from the filesystem
versus 30% when the old chats were pasted back in. For frog, phase 1 already gives an exact,
persisted store that costs no context: frog's memory layer can `put` summaries / facts as token
ids at the end of a chat and `get` them into the prompt by name at the start of the next. That is
still "pasting", but of a small, curated, exact file rather than old chats. The 90% number needs
phase 2, where the model itself asks for a file mid-generation and gets the exact tokens back.

## Phase 2 design: the model reads and writes the store

Goal (the post's design, adapted): the model emits a tool call `get(name)` / `put(name, text)`;
compiled layers capture the name and text into GDN state as the model writes them, and at
`</tool_call>` the circuit prints the `<tool_response>` itself. Nothing on the host.

Circuit (from the repo's `docs/q35-store.md`):
- extra "chip" dims in the residual: 18 id bits of the current token (an input feature added to
  each embedding row) and an 18-dim id output code that makes the LM head emit any stored id;
  chip dims are cleared before the base layers, so Qwen's own computation is unchanged;
- extra layers (the repo uses 16 compiled layers on the 4B, +100 heads, hidden 4224 → 4608) that
  hash/compare names bit by bit, keep registers (next free row, cursor) in one-hot GDN rows, and
  drive store heads with one-hot address groups (a key channel = sum of 2-bit one-hot groups
  − (n−1), so a 128-row table needs no 128-way one-hot);
- trace tokens: the post's CPU emits ADDR / VAL / STEP tokens per instruction, and its store
  emits "chip records" between the tool call and the response.

In tang, the decode loop replaces the post's external RvChat driver: a trace token is a token the
engine feeds through *only* the chip layers (a `gdn_step` + small MLP on the chip state), never
through the base layers or the KV cache, and never shown. The engine sees `</tool_call>` with a
store tool name (the tokenizer's ids), runs the chip machine's trace tokens as engine-internal
windows of ≤ 8 (exactly the windows phase 1 issues), then resumes base decoding with the circuit's
response tokens as forced input. So "tool-call tokens become store ops" twice over: phase 1 maps
them to `MemStore` calls in Rust (a few hundred lines in the serve tool dispatcher), phase 2 maps
them to compiled weights, and the two can be diffed token for token because both go through
`gdn_step`.

Effort and cost for Coder-Next (80B-A3B, 48 layers, GDN layers with 16 k / 32 v heads): the
compiler (a DSL → weights for one-hot GDN tables, bit compare, registers) is the big part; porting
the repo's `chipformer` design to Rust and to Coder-Next's shapes, plus the widened-layer loader
and the trace-token path in the decode loop, is roughly 4–8 weeks for one engineer, with the
risk in bf16/Q4 exactness of the chip rows (the repo is exact in bf16; tang's quantized
projections would need the chip rows kept in f32/bf16). Per-token cost: the store heads are
12 × 128² × ~6 flops ≈ 1.2 MFLOP; compiled layers of the repo's size (~800 extra dims, ~16 small
layers) are ~0.1 GFLOP, about 2% of Coder-Next's ~6 GFLOP active per token, paid on every token
unless gated; trace tokens add a few chip-only steps per store call, negligible.

## Test timings (CPU device, M-series Mac)

Debug defaults (128 files, 300 random ops): all 7 tests in 43 s wall, the two heavy ones in
parallel (~65 ms per window in debug). Full sizes in release
(`TANG_MEMSTORE_FILES=512 TANG_MEMSTORE_OPS=10000 cargo test --release -p tang-llm --lib memstore`):
512-file fill 3.4 s, 10,000-op workload 26.7 s (16,979 windows, 25,665 tokens), all pass.
