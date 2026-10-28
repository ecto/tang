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

- **Milestone 2** (a Rust compiler that emits these layers for given GDN shapes) is not
  started. The circuit is ~3.5k lines of python IR (`rv/core.py`, `rv/store_rv.py`,
  `q35/ir.py`, `q35/prims.py`) plus the lowering (`q35/emit.py`, ~300 lines). A first step
  would port only the lowering, fed an IR exported from python, checked weight-for-weight at
  4B shapes; Coder-Next (16/32/128, hidden 2048) also needs the widening the checkpoint
  does (≥ 1806 computer slots on top of 2048).
- **memstore 32-head layout.** `memstore.rs` assumes 48 v heads (`GDN_HV`, `v % 16` k-head
  mapping). Qwen3.5-4B and Coder-Next are 16/32 with *grouped* mapping (`vh / 2`); a 32-head
  layout must place directory/content on v heads 2j, 2j+1 of store k heads. Not added.
- **CUDA path**: keep bf16 rounding at the HF points, run the chip stack as its own GDN
  windows with persistent state, compacted GEMVs; conv state carry is needed (not in the
  author's snapshots).
- Only one trace (put/get/list) is checked; append/delete and the RV calc path are not.
