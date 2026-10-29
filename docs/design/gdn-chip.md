# GDN chip: compiled store layers in tang (phase 2 of zero-context compiled memory)

Phase 1 (`gdn-memstore.md`) keeps a file store in GDN heads but the *engine* drives every read
and write. Phase 2 lets the *model* do it: qwenputer (divagr.com/blog/qwenputer, MIT, Copyright
(c) 2026 Divyansh Agrawal; github.com/divagr18/qwenputer, branch `rv-store`,
`experimental/riscv`) compiles a circuit into the weights of stock Qwen3.5 GDN decoder layers.
`crates/tang-llm/src/flash/chip.rs` runs those layers in tang's CPU reference math.

## What the compiled layers compute

The Filesystem checkpoint (`divagr1925/Qwen3.5-4B-Qwenputer-Filesystem`) is 44 layers: 12 chip
layers (0..11), then Qwen3.5-4B's 32. Hidden 4608 = Qwen's 2560 dims + computer dims
2560..4607 (18 ballast dims carry the norms). GDN 16 k / 32 v heads, d_state 128, conv 4, MLP
9216. Every chip layer is a plain `linear_attention` layer:

- **Heads are tables.** 164 circuit heads (302 v heads). Keys are one-hot (a sum of 2-bit
  one-hot groups minus n−1), β = sigmoid(±128) (exactly 1 or 0), g = 0 (`A_log` = 0,
  `dt_bias` = −110: softplus underflows), so a write overwrites one 128-float row and nothing
  decays. A query reads one row. Value head x of k head j is v head 2j + x.
- **The store** (layout.json of the traces): registers L1 kh4; order L4 kh10–11; length L5
  kh10–11; directory L9 kh1–8 (1,024 hashed slots, slot s = head s/128 row s%128); next-chunk
  L10 kh0–3; content L9 kh9–15, L10 kh4–15, L11 kh1–13.
- **MLPs are gates.** Each SwiGLU unit is one product term (`gate = K·b`, `up = a/K`), giving
  the boolean/arithmetic logic between table reads.
- **Output.** After layer 11 the code dims (`code_dims`: family id + 16 payload bits) and the
  store's 18 id-code dims say which token the circuit owns the next position with (all zero:
  Qwen owns it). `chip::Decoder` is `RvChat.code_token`.
- Besides the store the same layers hold an RV32 core (ROM, RAM, registers) used for `calc`;
  its tables are initialised by `hf.inject` after the prompt (`Driver::start` takes them as
  `init` / `mask`).

## Milestone 1 (done): tang's math is what the weights assume

Ground truth: the reproduction on mew (HF modules, bf16, the author's fast driver) over
put("a.txt", 5 tokens), get, list after an 806-token prompt
(`/private/tmp/claude-501/qwenputer-traces`). tang loads layers 0..11 and the embedding from a
9 GB range-fetched slice of the checkpoint, the circuit's head inits and decoder table from the
python compiler (`core.build` + `core.snapshot`, CPU, no weights), and replays the driver.

- `Numerics::Bf16` (round where HF holds bf16): all 126 fed tokens identical (forced call
  tokens, 6 chip records, circuit-printed text incl. "ok", " alpha … epsilon", "a.txt"), and
  every store head's full 128×128 state **bitwise equal** after each op.
- `Numerics::F32` (plain f32): the same 126 tokens; states within 1.2e-4 (residues that bf16
  flushes to 0 stay as ~1e-5 in f32). The circuit is robust to the rounding but not bit-exact
  without it; a CUDA path should keep bf16 activations at the same points.

Tests: `cargo test -p tang-llm --lib chip` (the one-hot RAM layer test always runs;
`chip_milestone1_{bf16,f32}` are `#[ignore]`, run with `--ignored`, paths from
`TANG_CHIP_DIR` / `TANG_CHIP_TRACES`).

## Compiler (milestone 2, done): `crates/tang-chip`

A Rust port of qwenputer's lowering (`chipformer/q35/emit.py`, MIT, NOTICE in the crate), so
tang emits the chip layers itself for a given GDN shape. A crate of its own: it is a pure
compiler (IR in, weights out; serde + flate2 only) that build tools can use without tang-llm's
runtime; tang-llm depends on it to run what it emits.

- **IR** (`ir.rs`, after `q35/ir.py`): `Expr` (linear forms, dyadic coefficients), `Ffn`
  (units `coef · a · relu(b)`), `Head` (q/k channels with conv taps, value heads with values,
  decay selector `a`, output gates `z`, outputs, write gate β, `into`), `Graph` (dims, token
  dims, unit counts, big dims, keep, output dims). Read from JSON.
- **Allocation** (`alloc.rs`, `emit.py` `allocate`): residual slot sharing with subtract /
  clear handover units. Equal to python's on the store program (slots, subtracts, clears,
  final owners: `chip_alloc_matches_python`).
- **Lowering** (`emit.rs`, `emit.py` `emit_layers`, `embedding_rows`, `merge.py`
  `pick_hidden`): every rule cites its python line. Output: per layer the 14 HF tensors, the
  matrices as sorted nonzero entries (`Sparse`; tang-llm runs them as CSR), plus the decoder's
  code / id-code dims and embedding rows from IR token features.
- **The store program**: not transcribed. `scripts/chip/export_ir.py` serialises the python
  `core.build(T4B, Text, store.Config)` graph (the whole Filesystem circuit: RV32 core and the
  object store, put/get/list/delete/append; 3,753 dims, 164 heads, 47,532 units, 12 layers);
  it is bundled gzip'd (600 KB, `tang_chip::store_program()`). delete/append are emitted (and
  bitwise equal, below) but only put/get/list are exercised by a trace.

Parity (`cargo test -p tang-chip -- --ignored`, `TANG_CHIP_DIR`): emitted at Qwen3.5-4B shapes
(`ChipShape::QWEN35_4B`: base 2560, 2048 chip dims, 18 ballast, 16/32/128, MLP 9216) all 168
tensors of layers 0..11 equal the checkpoint's **bitwise** in bf16, every element (2.44 G,
1.24 M nonzero), padding included. Run through chip.rs (`chip_milestone2_emitted_bf16`,
ignored) the emitted layers give the milestone-1 result: the 806-token prompt, injection, and
all 126 fed tokens identical, every store head bitwise equal after each op.

Retarget (`chip_retarget_coder_next`, always runs: ~30 s debug, 2.3 s release): emitted for
base 2048, 16 k / 32 v / d 128, chip tokens at 262144. put/get/list from zero state (the store
needs neither the prompt nor the injected RV tables: checked on the checkpoint) reproduce the
trace's tokens and store states bitwise; no weight reads or writes a host dim, the host part
of the residual is unchanged, and v heads without value rows stay zero.

**Coder-Next width.** The circuit needs 1,806 slots + tau + ballast. `pick_hidden(2048, 1807)`
gives hidden **3968** (= 62 · 4³), i.e. **1,920 chip dims**: 62 ballast, tau, 1,806 slots, 51
spare (4B: 2,048 chip dims, 18 ballast). `m · B² / hidden` must be a power of 4, which is what
fixes these sizes.

What differs from the python: weights come out sparse (python emits dense f32 then casts);
torch's `t[i] += x` (f64 → f32 then f32 add) and double → bf16 (via f32) are mirrored; the
`clearable` restriction and the exactness evaluator (`ir.evaluate`, `kernel_check`) are not
ported (the store build passes `clearable = None`); text-token rows (`rv/merge.py` `text_rows`:
UTF-8 bytes, the store's 18 id bits, tool-call flags), the final-norm gains and Qwen muting are
not ported either, nor graph construction.

Retarget limits: the graph is scheduled for `T4B` (≤ 16 k heads per layer, 2 v heads per k
head, ≤ 9216 units per MLP); a host with fewer heads or a narrower MLP needs python to
re-schedule (or a port of the builder). The chip layers are their own dense-MLP GDN layers, so
Coder-Next's MoE layers do not have to host them. Text features are those of Qwen3.5's
tokenizer: for Coder-Next the IR must be exported from `core.build` with its tokenizer (the
store's id code is the token id's bits; printable/echo tables are tokenizer ids), and its text
rows built; the retarget test therefore keeps Qwen3.5 text ids and moves only the chip tokens.

Left for engine/CUDA integration: text rows for the host vocabulary in the widened embedding
(the host's own rows plus ballast and features), a chip-layer window in the decode loop with
its own GDN/conv state, the decoder on the residual after the chip stack, compacted (CSR or
block) GEMVs with bf16 rounding at the HF points, and the RV tables' injection for `calc`.

## Engine (milestone 3): the chip window on CUDA

`crates/tang-llm/src/flash/chip_gpu.rs` (cuda), its packing in `chip.rs` (`ChipPack`,
`PackedCsr`, `PackedDecoder`, `pick_next`, `RetargetFixture`).

**What's on the device.** Per layer: the four input projections as one stacked CSR
(`qkv | z | a | b`), out, gate|up stacked, down. The conv weights, dt_bias, −exp(A_log), the
norm gains, the embedding rows of the tokens the window is fed (with a token → row map), and
the decoder table (families as `[kind, a, b]`, printable, code / id-code dims). The blocks are
~0.05% dense, so each row is a CSR row summed by one thread. Dense or block GEMVs would read
2.4 G weights per token for ~0.33 M nonzeros. Kernels: `chip_rmsnorm`,
`chip_csr` (optionally fused `x = bf(x + bf(Wy))`), `chip_conv` (conv, SiLU, history shift),
`chip_l2`, `chip_gdn` (recurrence + gated norm, one block per v head), `chip_act`,
`chip_pick` (decoder + next-token rule on the device), `chip_embed`.

**Why not `gdn_step_body`.** Its state is interleaved across heads (`[r][GDN_HV][128]`), it
uses FMA and split reductions, and tang-compute's generic device `gdn_step` is hard-wired to
16 k / 48 v. chip.rs's math is `reference::gdn_head_step` (sequential sums, separate
roundings). `chip_gdn` runs that loop per value column in the CPU's order, with
`__fmul_rn`/`__fadd_rn` (no contraction) and exp/log in double. The circuit's arithmetic is
exact, so this is bitwise by construction rather than "close". The Coder engine's own GDN path
uses its runtime `Geometry` (`gdn_hv` from the GGUF, grouped `h / r`); the `GDN_HV = 48`
constants are defaults only.

**bf16 rounding points** (chip.rs `Layer::step`, Numerics::Bf16, the same on the device):
both RMSNorm outputs; every projection output; the conv sum and its SiLU; β; the GDN core
output; the gated norm's normed value, its gain product and the SiLU(z) product; the out
projection and the residual after it; gate, up, SiLU(gate) and the product; down and the
residual after it. CSR rows sum in column order from −0.0 (Rust's f32 `Sum`). The one
reordered reduction is the RMSNorm sum of squares (a block tree). On every chip-window token
it is `m · 2^80` plus terms below its half-ulp, so the order cannot change it.

**Window mechanics.** `ChipGpu::op(slot, forced)` writes a control block (cur token, queue of
forced tokens, fed log). A window is 8 tokens of `embed → 12 layers → pick`, captured once per
slot as a CUDA graph. Every kernel returns at once when the op is done or flagged, and the host
reads the control block between windows. `pick` is `Driver::op`'s rule: a chip token is
followed by the circuit's; a text token by the circuit's when it owns the position, else by the
next forced token; an empty queue ends the op. Errors (no row, no token for a code, runaway)
are flagged in the block and raised on the host.

**Kept out of the host paths.** `Coder::chip_op(slot, tokens)` (coder.rs, next to
`store_op_at`) only calls `ChipGpu::op`. It never enqueues the Coder window (`window_mode`,
`prefill_slot`, `decode_batch`), so chip tokens never reach the base layers, the head's
sampling, the KV cache, the MTP layer or `slot_tokens`. `store_op_at` takes the chip path when
`load_chip` was given a `ChipCodec`; otherwise it keeps the memstore windows.

**State per slot.** `ChipGpu` keeps per-slot conv history `[3][conv_dim]` and recurrent state
`[v_heads][128][128]` per layer (chip.rs `ChipState` layout), allocated zeroed on first use.
`free_slot` drops a slot's state; `reset` zeroes all of it, as it zeroes the memstore heads.

**Memory at Coder-Next width** (hidden 3968, 12 layers): weights = CSR (ptr + idx + val) +
vectors; per slot 12 × (3 · 8192 + 32 · 128²) floats = 25.5 MB. Measured numbers below.

**Verification** (`chip_gpu_put_get_list_matches_cpu`, `TANG_REQUIRE_CUDA=1`):
1. *Shadow + poison.* Before each token the scratch is filled with NaN. After each layer the
   residual, conv history and recurrent state are compared bitwise with chip.rs's layer step
   from the same input. A mismatch is reported as op / token / layer and resynced. The
   decoded token is checked against `Decoder::token`.
2. *Production.* Graph windows on a fresh slot: every fed token equals the trace's. After each
   op, every layer's full state is bitwise equal to the CPU `Driver`'s, and the store heads to
   the trace's.
3. *In the Coder* (`chip_gpu_in_coder`, `TANG_LLM_TEST_GGUF`, any qwen3next GGUF as the host).
   chip_op on a slot with a prompt leaves its tokens unchanged, and its greedy continuation
   equals a twin slot's. `store_op_at` with a table codec returns Put / Got(ids) / Listed.

**Measured numbers.** RESULTS_PENDING

**What remains.** Real Coder-Next weights and a tokenizer re-export: the IR from
`core.build` with Coder-Next's tokenizer, and text rows for the host vocabulary in the widened
embedding. A real `ChipCodec`: tool-call text → tokens, printed answer → `OpOut`; the test's
`TableCodec` is the fixture's. The serve route (`/v1/memory/op` → `store_op_at`, and Sessions
export/import of the chip state, which `store_heads` does not cover). The sampler handoff: when
the model emits a tool call, the decode loop must switch the slot to the chip window and feed
the printed answer back through the base layers as ordinary text. The RV `calc` tables'
injection.

## Qwen3.5-4B support

`reference.rs` targets GGUF Qwen3-Next. Qwen3.5's GDN is the same math (grouped k heads,
SiLU output gate, L2 q/k with eps 1e-6, q/√d, decay-then-delta), so the recurrence is shared
(`reference::gdn_head_step`). What chip.rs adds: HF safetensors loading, separate
`in_proj_a/b`, the `1 + w` RMSNorm convention, dense SwiGLU MLP, per-token decode with conv
history, block-compacted weights, and bf16 rounding points. Qwen3.5's gated full attention
(every 4th layer) is not needed by the chip layers and is not ported.

## Trace tokens stay out of Qwen

In the author's driver chip tokens (ids ≥ 248320) run *only* through the chip layers with
their own cache; Qwen's 32 layers and its KV cache see only text tokens. In tang's decode loop
this is an engine-internal window: on a position the circuit owns, the engine feeds the
circuit's token to the chip-layer stack (its own GDN state, like memserve's store windows)
and, for chip records, never runs the base layers or appends KV. Circuit-printed text (the
tool response) is fed to both, like any generated text. Nothing reaches the sampler's
context except the printed response.

## Compute per token (measured)

The chip layers' nonzero blocks are 331.8 M multiply-adds per token over 12 layers (dense
would be 2.4 G); plus 302 active 128×128 GDN heads. On the Mac CPU reference (16 threads,
f32) a decode step through all 12 layers takes ~13 ms. For scale, Qwen3.5-4B is ~4 G MACs per
token, so the store adds ~8% with compacted weights.

## Not done

- **Circuit builder in Rust**: the store program is python's graph, exported; building it (and
  re-scheduling for other targets or tokenizers) in Rust is not done (see "Compiler").
- **memstore 32-head layout.** `memstore.rs` assumes 48 v heads (`GDN_HV`, `v % 16` k-head
  mapping). Qwen3.5-4B and Coder-Next are 16/32 with *grouped* mapping (`vh / 2`); a 32-head
  layout must place directory/content on v heads 2j, 2j+1 of store k heads. Not added.
- **CUDA path**: keep bf16 rounding at the HF points, run the chip stack as its own GDN
  windows with persistent state, compacted GEMVs; conv state carry is needed (not in the
  author's snapshots).
- Only one trace (put/get/list) is checked; append/delete and the RV calc path are not.
